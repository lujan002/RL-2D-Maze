# RL-2D-Maze
Custom gymnasium environment for reinforcement learning to solve a 2D maze with continuous action space. 

## Overview
Several strategies for engineering an optimal reward function were attempted, ultimately the best performance came from a reward function that gave a large reward for reaching the goal, a small reward for steping towards the goal, and a large penalty for stepping away from the goal. The observation space consisted of the agent's relative position to the optimal trajectory, distance of eight simulated lidar rays, and the relative position of the coordinates of a few of the agent's recent past states. 

More specifically, the optimal trajectory is calculated with the astar algorithm and several green "breadcrumb" dots in the environment mark this path to the goal. The agent's relative position to these breadcrumbs, along with the last few state coordinates, is calculated as a gaussian-normalized position, where the difference in position can be thought of as the x-input to a gaussian function of e^-(lamda*x^2) which always returns a value of 0-1 and where lamda is a emperically determined constant. In general, training RL agents is much more effective when the reward and observation signals are normalized in this way. 

## Project Organization
The project is broken down into four main scripts: A script for creating the custom gymnasium environment (continuous_maze_env.py), a script for manually testing, troubleshooting, and moving the agent manually (test_continuous_maze.py), a script for training (train_continuous_maze.py), and finally a script for evaluating the trained model (evalutate_continuous_maze.py)

Continuous_maze_env refers to a maze.py script, which was adapted from this repo (https://github.com/Turidus/Python-Maze) with minor edits to make it compatible with this environment. This script handles the generation of a random maze of any size. 

Another script, astar_pathfinding.py, calculates the shortest distance from the agent starting position to the goal position. This algorithm is well studied and there are many resources online if you wish to familiarize yourself with it. 

## Getting Started
### Create a new conda environment with all dependencies from 2D_maze.yml 
    conda env create -f 2D_maze.yml
    conda activate 2D_maze.yml
Make sure to change the interpreter path of your IDE to ~/anaconda3/envs/2D_maze/bin/python.

To generate the maze environment and move the agent manually, run continous_maze_env.py followed by test_continuous_maze.py
If you wish to test the agent's ability to reach the goal in a more simple environment (just an empty space without the maze), you can replace instances of the "generate_maze" method with "generate_empty_maze". 

To train the model, run train_continuous_maze.py. Hyperparameters can be configured in ppo_config.yaml. I have also set up a training script for the popular "lunar_lander" gymnasium environment, which may prove a useful tool to benchmark your model.

## Remaining Work
The agent is able to reach the goal some of the time, but it is not clear if this is due to random luck or if the agent actually knows to head towards the goal, maybe a mix of both. The best reward function I found had good sucess in getting the agent to explore, but it is not uncommon that agent goes in a complete opposite direction of the goal. More work needs to be done to refine the reward function and observations. 

There is also a discrepency between the mean reward calculated in the training (from stable baselines) and the mean reward calculated in my custom evaluation script. This means that I have a lack of confidence that the agent I am watching being evaluated is actually representative of the agent I have trained. I did observe that when setting "deterministic" = true, the agent gets stuck in a state of running into the wall over and over. When "deterministic" = false, the agent was seen to explore, albeit quite randomly. This points to a problem with the agent's policy, as it actually performs better when some randomness is injected into it's policy to help it get it over these perceived local maxima.

This project is by no means a completed work, much remains to be improved, but I hope the work I have done up until now may be a good foundation for future development.








