# RL-2D-Maze
Custom gymnasium environment for reinforcement learning to solve a 2D maze with continuous action space. 

https://github.com/user-attachments/assets/7f6ae416-db74-44ad-a865-09c8be3df37c

**Important: The above agent was acheived by training in the "best-agent" branch for 800k timesteps. The main branch contains the most recent experiments, however, this older version has the best agent performance yet.**

## Overview
Several strategies for engineering an optimal reward function were attempted, ultimately the best performance came from a reward function that gave a large reward for reaching the goal, a small reward for steping towards the goal if the new position had not been previously reached, and a small penalty for stepping away from the goal if the new position had not been reached. This was coupled with a observation space that consisted of the agent's relative position to the optimal path to the goal, the distance of eight simulated lidar rays, and the relative position from the agent's current position to a few of the agent's most recent positions. 

## The Environment
This [repo](https://github.com/Turidus/Python-Maze) was adapted to generate a random grid-based maze. Several options exist for controling the size of the maze, number of loops, dead ends, etc. The agent is a red rectancle with eight evenly-spaced blue lidar rays eminating from it. The goal is a large green dot. The agent and the goal are spawned at a random location in the maze as long as that location is not a wall and they are not too close. 

More specifically, the optimal trajectory is calculated with the astar algorithm and several green "breadcrumb" dots in the environment mark this path to the goal. When the agent hits a breadcrumb, the breadcrumb disapears from the environment and optionally receives a small reward. The observation of the agent's relative position to the nearest breadcrumb and its most recent positions is calculated as a gaussian-normalized position, where the difference in position can be thought of as the x-input to a gaussian function of $f(x) = e^{-\lambda x^2}$ which always returns a value of 0-1 and where $\lambda$ is a emperically determined constant. In general, training RL agents is much more effective when the reward and observation signals are normalized in this way. 

Other reward-observation combos considered were a goal proximity reward-observation combo, a small timestep penalty, a small wall collision penalty, a visited cell reward-observation combo, separate observations for the relative position to the goal and the breadcrumbs, a reward for longer lidar distances (to encourage agent to be far from the wall),  

## Project Organization
The project is broken down into four main scripts: A script for creating the custom gymnasium environment: `continuous_maze_env.py`, a script for manually testing, troubleshooting, and moving the agent manually: `test_continuous_maze.py`, a script for training: `train_continuous_maze.py`, and finally a script for evaluating the trained model: `evalutate_continuous_maze.py`.

All records of experimental reward-observation combos are saved in `continuous_maze_env.py`. 

Another script, `astar_pathfinding.py`, calculates the shortest distance from the agent starting position to the goal position using the "A*" search algorithm. This algorithm is well documented online if one wishes to familiarize themselves with this topic. 

## Getting Started
### Create a new conda environment with all dependencies from 2D_maze.yml 
    conda env create -f 2D_maze.yml
    conda activate 2D_maze.yml
Make sure to change the interpreter path of your IDE to "~/anaconda3/envs/2D_maze/bin/python".

To generate the maze environment and move the agent manually, run `continous_maze_env.py` followed by `test_continuous_maze.py`
To test the agent's ability to reach the goal in a simple environment (just an empty space without the maze), replace instances of the "generate_maze" method with "generate_empty_maze". 

To train the model, run `train_continuous_maze.py`. Hyperparameters can be configured in `ppo_config.yaml`. I have also set up a training script for the popular "lunar_lander" gymnasium environment, (`train_lunar_lander.py`), which may prove a useful tool to benchmark your model.

## Remaining Work
The agent is able to reach the goal some attempts, but it is not clear if this is due to random luck or if the agent actually knows to head towards the goal, maybe a mix of both. The best reward function I found had good sucess in getting the agent to explore, but it is not uncommon that agent goes in a complete opposite direction of the goal. The agent also has a tendency to get stuck running into walls after initial sucess exploring the environment. It is not clear what is causing this behavior. 

There is also a discrepency between the mean reward calculated in the training (from stable baselines) and the mean reward calculated in my custom evaluation script. This means that I have a lack of confidence that the agent I am watching being evaluated is actually representative of the agent I have trained. I did observe that when setting "deterministic" = true, the agent gets stuck in a state of running into the wall over and over. When "deterministic" = false, the agent was seen to explore, albeit quite randomly. This points to a problem with the agent's policy, as it actually performs better when some randomness is injected into it's policy to help it get it over these perceived local maxima.

This project is by no means a completed work, much work remains to be done to refine the rewards, observations, and troubleshoot the root cause of these issues, but hopefully this work may be a good foundation for future development.








