# Monte Carlo Tree Search for LLM Reasoning

A Python implementation of Monte Carlo Tree Search (MCTS) to guide Large Language Models in solving math problems step-by-step.

## Overview

This project uses MCTS to explore and evaluate different reasoning paths for solving mathematical problems. It combines:

1. A generative LLM (Llama-3.2-1B-Instruct) to create reasoning steps
2. A specialized math reasoning evaluation model (math-shepherd-mistral-7b-prm) to score the quality of steps
3. MCTS algorithm to intelligently search the reasoning space

## How It Works

The implementation follows the classic MCTS algorithm with four phases:

1. **Selection**: Choose the most promising node in the tree using Upper Confidence Bound (UCB)
2. **Expansion**: Generate new reasoning steps using the LLM
3. **Simulation**: Evaluate the quality of reasoning using the reward model
4. **Backpropagation**: Update the statistics in the tree

Each node in the tree represents a state in the reasoning process, with reasoning steps as actions connecting nodes.

## Features

- Step-by-step mathematical reasoning
- Tree visualization to track search progress
- Weighted evaluation that prioritizes correct final answers
- Configurable exploration vs. exploitation balance
- Special token system for targeted step evaluation

## Installation

```bash
pip install torch transformers tqdm
```

## Usage

```python
from mcts_llm_reasoning import solve_problem

# Define a math problem
problem = "Chickens and rabbits are in a barn. There are 20 heads and 56 legs in total. How many chickens and how many rabbits are there?"

# Solve it using MCTS
solution = solve_problem(problem, iterations=10)
print(solution)
```

## Configuration

The MCTS algorithm can be configured with:

- `max_iterations`: Number of MCTS iterations to run
- `max_depth`: Maximum depth of the search tree
- `exploration_weight`: Controls exploration vs. exploitation tradeoff

## Models Used

- **Generator Model**: meta-llama/Llama-3.2-1B-Instruct
- **Reward Model**: peiyi9979/math-shepherd-mistral-7b-prm

## Technical Details

The implementation uses a special token system ('ки') to mark the end of reasoning steps for targeted evaluation by the reward model. The reward model calculates scores for each step, with higher weight given to the final answer.

## Example Output

The algorithm produces both step-by-step reasoning and visualization of the search tree:

```
Problem: Chickens and rabbits are in a barn. There are 20 heads and 56 legs in total. How many chickens and how many rabbits are there?

--- Full Best Solution Path ---
Step 1: Step 1: Define the variables for the number of chickens and rabbits. Let C represent the number of chickens and R represent the number of rabbits. The equation based on the given information is: C + R = 20 and 2C + 4R = 56.
[visits=10, avg_value=0.7022]

Step 2: Step 2: To solve the system of equations, we can first solve one of the equations for one variable. Let's solve the first equation for C: C = 20 - R. Then substitute this expression for C into the second equation. The resulting equation is: 2(20 - R) + 4R = 56. Expand the equation: 40 - 2R + 4R = 56. Combine like terms: 40 + 2R = 56. Subtract 40 from both sides: 2R = 16. Divide by 2: R = 8. Now that we know the number of rabbits, we can substitute R = 8 into the first equation to solve for C. C = 20 - R = 20 - 8 = 12. The number of chickens is 12 and the number of rabbits is 8.
[visits=4, avg_value=0.7746]

Step 3: Step 3: The number of chickens is 12 and the number of rabbits is 8.
[visits=1, avg_value=0.8133]
```

The search tree visualization helps track the algorithm's exploration:

```
===== SEARCH TREE STATUS =====
ROOT [visits=28, avg_value=0.6375]
└── Problem: Chickens and rabbits are in a barn. There are 20 h...
├── Step 1: Define the variables for the number of chi...
│   [visits=10, avg_value=0.7022]
│   ├── Step 2: To solve the system of equations, we can f...
│   │   [visits=4, avg_value=0.7746]
│   │   ├── Step 3: The number of chickens is 12 and the numbe...
│   │   │   [visits=1, avg_value=0.8133]
│   │   ├── Step 3: Write a system of equations using the vari...
│   │   │   [visits=1, avg_value=0.8023]
│   │   └── Step 3: Since there are 12 chickens and 8 rabbits,...
│   │       [visits=1, avg_value=0.7176]
│   ├── Step 2: Solve the system of equations using the me...
│   │   [visits=4, avg_value=0.7255]
│   │   ├── Step 3: Since the problem asks for the number of c...
│   │   │   [visits=1, avg_value=0.7660]
│   │   ├── Step 3: Calculate the total number of chickens and...
│   │   │   [visits=1, avg_value=0.7223]
│   │   └── Step 3: Write the solution in the requested format...
│   │       [visits=1, avg_value=0.7059]
│   └── Step 2: Multiply both sides of the first equation ...
│       [visits=1, avg_value=0.5219]
├── Step 1: Let's assign variables to the unknown quan...
│   [visits=10, avg_value=0.6564]
│   ├── Step 2: To solve the system of equations, we can u...
│   │   [visits=4, avg_value=0.7313]
│   │   ├── Step 3: Substitute the expression for C into the s...
│   │   │   [visits=1, avg_value=0.8250]
│   │   ├── Step 3: Substitute the expression for C into the s...
│   │   │   [visits=1, avg_value=0.7340]
│   │   └── Step 3: Substitute the expression for C into the s...
│   │       [visits=1, avg_value=0.7176]
│   ├── Step 2: To solve the system of equations, we can u...
│   │   [visits=1, avg_value=0.6270]
│   └── Step 2: Multiply the second equation by 2 to make ...
│       [visits=4, avg_value=0.5536]
│       ├── Step 3: Multiply the first equation by 8 and the s...
│       │   [visits=1, avg_value=0.6582]
│       ├── Step 3: Multiply the first equation by 8 to make t...
│       │   [visits=1, avg_value=0.5297]
│       └── Step 3: Subtract the first equation from the modif...
│           [visits=1, avg_value=0.3778]
└── Step 1: Let C represent the number of chickens and...
    [visits=8, avg_value=0.5330]
    ├── Step 2: To solve this system of equations, we can ...
    │   [visits=2, avg_value=0.8598]
    ├── Step 2: Solve the system of equations. Multiply bo...
    │   [visits=1, avg_value=0.4052]
    └── Step 2: Solve the system of equations by multiplyi...
        [visits=4, avg_value=0.3152]
        ├── Step 3: Multiply the equation 9C + 19R = 120 by 5 ...
        │   [visits=1, avg_value=0.3138]
        ├── Step 3: Since we want to find the number of chicke...
        │   [visits=1, avg_value=0.2557]
        └── Step 3: Now, let's solve the new system of equatio...
            [visits=1, avg_value=0.2389]
```