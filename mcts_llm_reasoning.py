import torch
import math
import random
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class MCTSNode:
    def __init__(self, state: str, parent=None, action=None, exploration_weight=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.exploration_weight = exploration_weight

    def is_fully_expanded(self) -> bool:
        """Check if node has been expanded with children."""
        return bool(self.children)
    
    def is_terminal(self) -> bool:
        """For LLM reasoning, we consider a node terminal if it seems to be a final answer."""
        # This is a simple heuristic, can be improved based on specific use case
        return "answer is" in self.state.lower() or "therefore" in self.state.lower()
    
    def get_ucb(self, parent_visits: int) -> float:
        """Calculate the UCB score for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = self.exploration_weight * math.sqrt(2 * math.log(parent_visits) / self.visits)
        return exploitation + exploration

class MCTS:
    def __init__(self, problem: str, generator_model_name: str, reward_model_name: str, 
                 max_iterations: int = 100, max_depth: int = 5, exploration_weight: float = 1.0):
        self.problem = problem
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        
        # Load generator model (Qwen2.5-Math)
        print("Loading generator model...")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load reward model (math-shepherd)
        # Load reward model (math-shepherd)
        print("Loading reward model...")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Initialize root node with the problem statement
        self.root = MCTSNode(state=problem, exploration_weight=exploration_weight)
    def visualize_tree(self, node=None, depth=0, max_depth=3, prefix=""):
        """Visualize the current state of the MCTS tree with rewards and visit counts."""
        if node is None:
            node = self.root
            print("\n===== SEARCH TREE STATUS =====")
        
        # For the root node (problem statement)
        if depth == 0:
            avg_value = node.value / max(node.visits, 1)
            print(f"ROOT [visits={node.visits}, avg_value={avg_value:.4f}]")
            print(f"└── Problem: {self.problem[:50]}...")
        
        # Stop if we've reached max depth
        if depth >= max_depth:
            if node.children:
                print(f"{prefix}    └── ... ({len(node.children)} more nodes)")
            return
        
        # Sort children by their UCB score for better visualization
        children = sorted(node.children, key=lambda c: c.value / max(c.visits, 1), reverse=True)
        
        # Print each child node
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            avg_value = child.value / max(child.visits, 1)
            
            # Extract the step content (first 50 chars)
            action_str = child.action[:50] + "..." if child.action and len(child.action) > 50 else child.action
            
            # Create the line for this node
            if is_last:
                print(f"{prefix}└── {action_str}")
                print(f"{prefix}    [visits={child.visits}, avg_value={avg_value:.4f}]")
                next_prefix = prefix + "    "
            else:
                print(f"{prefix}├── {action_str}")
                print(f"{prefix}│   [visits={child.visits}, avg_value={avg_value:.4f}]")
                next_prefix = prefix + "│   "
            
            # Recursively visualize children
            self.visualize_tree(child, depth + 1, max_depth, next_prefix)


    def select(self, node: MCTSNode) -> MCTSNode:
        """Select the most promising node based on UCB."""
        current = node
        depth = 0
        
        # Traverse down the tree until we reach a leaf node or max depth
        while current.is_fully_expanded() and not current.is_terminal() and depth < self.max_depth:
            best_score = -float('inf')
            best_child = None
            
            for child in current.children:
                ucb_score = child.get_ucb(current.visits)
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_child = child
            
            if best_child is None:
                break
                
            current = best_child
            depth += 1
            
        return current
    
    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        """Expand the node by generating 3 new reasoning steps using the LLM."""
        expanded_nodes = []
        
        # Generate 3 different children
        for i in range(3):
            # Determine the step number
            step_num = 1
            if node != self.root:
                # Count existing steps by looking for "Step" in the state
                steps = [s for s in node.state.split("\n") if s.strip().startswith("Step")]
                step_num = len(steps) + 1
            
            # Format the prompt to generate the next step without seeing the "ки" token
            if node == self.root:
                # First step
                prompt = f"""Solve this math problem step by step. Generate ONLY Step 1, nothing more:
                
                Problem: {self.problem}
                
                Step 1:"""
            else:
                # Remove "ки" tokens from previous steps for clean input to model
                clean_state = node.state.replace(" ки", "")
                
                prompt = f"""Continue solving this math problem. Generate ONLY Step {step_num}, nothing more:
                
                Problem: {self.problem}
                
                Previous reasoning:
                {clean_state}
                
                Step {step_num}:"""
            
            # Generate the next step with stricter constraints
            inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.generator_model.device)
            
            # Stop sequences to prevent multiple steps
            stop_sequences = [f"\nStep {step_num+1}", "\nStep", "\n\n", "Step"]
            
            output = self.generator_model.generate(
                **inputs,
                max_new_tokens=200,  # Reduced to limit length
                temperature= 0.8,
                top_p=0.9,
                num_return_sequences=1,
                do_sample=True
            )
            
            # Extract the newly generated text
            continuation = self.generator_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Post-process to ensure only one step is included
            for stop_seq in stop_sequences:
                if stop_seq in continuation:
                    continuation = continuation.split(stop_seq)[0]
            
            # Format with step number if not already formatted
            if not continuation.strip().startswith(f"Step {step_num}:"):
                step_text = f"Step {step_num}: {continuation.strip()}"
            else:
                step_text = continuation.strip()
            
            # Add the special token for scoring
            step_with_token = f"{step_text} ки"
            
            # Create the new state by appending to current state
            new_state = node.state
            if new_state and not new_state.endswith("\n"):
                new_state += "\n"
            new_state += step_with_token
            
            # Create a new node with the updated state
            child = MCTSNode(state=new_state, parent=node, action=step_text, exploration_weight=node.exploration_weight)
            node.children.append(child)
            expanded_nodes.append(child)
            
            print(f"Generated child {i+1}: {step_text}")
        
        return expanded_nodes

    
    
    def simulate(self, node: MCTSNode) -> float:
        """Evaluate the node using the math-shepherd reward model with step-by-step scoring."""
        # Define token constants if not already defined as class attributes
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'
        
        # Format the input for the reward model
        input_for_prm = f"{self.problem} {node.state}"
        input_id = torch.tensor([self.reward_tokenizer.encode(input_for_prm)]).to(self.reward_model.device)
        
        # Get token IDs for evaluation
        candidate_tokens = self.reward_tokenizer.encode(f"{good_token} {bad_token}")[1:]
        step_tag_id = self.reward_tokenizer.encode(f"{step_tag}")[-1]
        
        with torch.no_grad():
            logits = self.reward_model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]
            step_positions = (input_id == step_tag_id).nonzero(as_tuple=True)
            
            # If no step tags found, return a low score
            if len(step_positions[1]) == 0:
                return 0.0
            
            # Get scores at step positions
            step_scores = scores[0, step_positions[1]]
            
            # Average of all steps, with more weight on the last step (final answer)
            if len(step_scores) > 1:
                # Weight the final step more heavily (0.7 for final step, 0.3 for average of others)
                final_step_score = step_scores[-1].item()
                earlier_steps_avg = torch.mean(step_scores[:-1]).item()
                reward = 0.7 * final_step_score + 0.3 * earlier_steps_avg
            else:
                reward = step_scores[0].item()
                
            return reward
    
    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Update the statistics of all nodes in the path from the node to the root."""
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent
    def get_best_solution(self) -> str:
        """Return the full solution by following the best path through the tree."""
        print("\n--- Full Best Solution Path ---")
        
        # Start at the root
        current = self.root
        solution_path = []
        depth = 0
        
        # Format the problem statement
        solution_path.append(f"Problem: {self.problem}\n")
        
        # Follow the best path through the tree
        while current.children and depth < self.max_depth:
            # Find the child with the highest average value
            best_child = max(current.children, key=lambda c: c.value / max(c.visits, 1))
            
            # Add this step to the solution path
            if best_child.action:
                solution_path.append(best_child.action)
            
            # Print current step info
            avg_value = best_child.value / max(best_child.visits, 1)
            print(f"Step {depth+1}: {best_child.action}")
            print(f"[visits={best_child.visits}, avg_value={avg_value:.4f}]\n")
            
            # Move to the best child
            current = best_child
            depth += 1
            
            # If we've reached a terminal node, stop
            if current.is_terminal():
                break
        
        # Join all steps into a single solution string
        full_solution = "\n".join(solution_path)
        
        return full_solution
    def search(self) -> str:
        """Run the MCTS algorithm and return the best solution."""
        for iteration in tqdm(range(self.max_iterations), desc="MCTS Iterations"):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Selection
            print("Phase: Selection")
            leaf = self.select(self.root)
            print(f"Selected Node: {leaf.state[:100]}... (Visits: {leaf.visits}, Value: {leaf.value:.4f})")

            # Expansion (if not terminal)
            if not leaf.is_terminal():
                print("Phase: Expansion")
                expanded_nodes = self.expand(leaf)
                print(f"Expanded {len(expanded_nodes)} child nodes")
                
                # Process each expanded node
                for i, child_node in enumerate(expanded_nodes):
                    # Simulation for each child
                    print(f"\nSimulating child {i+1}")
                    reward = self.simulate(child_node)
                    print(f"Child {i+1} Reward: {reward:.4f}")
                    
                    # Backpropagation for each child
                    print(f"Backpropagating child {i+1}")
                    self.backpropagate(child_node, reward)
            else:
                # If terminal, just simulate and backpropagate the leaf
                print("Node is terminal, proceeding to simulation")
                reward = self.simulate(leaf)
                print(f"Simulated Reward: {reward:.4f}")
                
                print("Phase: Backpropagation")
                self.backpropagate(leaf, reward)
                
            # Show tree status after each iteration
            self.visualize_tree()

        # Return the complete solution by following the best path
        return self.get_best_solution()
    
    def get_solution_trace(self) -> List[Dict]:
        """Get the trace of the best solution path."""
        best_node = max(self.root.children, key=lambda c: c.value / max(c.visits, 1)) if self.root.children else self.root
        
        trace = []
        current = best_node
        while current:
            trace.append({
                "state": current.state,
                "visits": current.visits,
                "value": current.value,
                "ucb": current.get_ucb(current.parent.visits) if current.parent else 0
            })
            current = current.parent
        
        return list(reversed(trace))

def solve_problem(problem: str, iterations: int = 10) -> str:
    """Main function to solve a math problem using MCTS with LLMs."""
    mcts = MCTS(
        problem=problem,
        generator_model_name="meta-llama/Llama-3.2-1B-Instruct",
        reward_model_name="peiyi9979/math-shepherd-mistral-7b-prm",
        max_iterations=iterations
    )
    
    solution = mcts.search()
    return solution

if __name__ == "__main__":
    # Example usage
    problem = "Chickens and rabbits are in a barn. There are 20 heads and 56 legs in total. How many chickens and how many rabbits are there?"
    solution = solve_problem(problem)
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")
    
# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
# import torch

# good_token = '+'
# bad_token = '-'
# step_tag = 'ки'

# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
# candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
# step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()

# question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
# output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
# output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

# for output in [output1, output2]:
#     input_for_prm = f"{question} {output}"
#     input_id = torch.tensor([tokenizer.encode(input_for_prm)])

#     with torch.no_grad():
#         logits = model(input_id).logits[:,:,candidate_tokens]
#         scores = logits.softmax(dim=-1)[:,:,0] 
#         step_scores = scores[input_id == step_tag_id]
#         print(step_scores)
        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
