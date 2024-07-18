import gymnasium as gym
import continuous_maze_env  # Ensure this imports your environment file
import pygame
import time 
import matplotlib.pyplot as plt
from Maze import Maze
from gymnasium.wrappers import FlattenObservation
import sys
import numpy as np
sys.path.append('Users\20Jan\TU Graz Code\2D_maze_project')

# Initialize lists to store rewards and timesteps
all_rewards = []
episode_rewards = []
cumulative_reward = 0

# Define the control keys
TRANSLATION_KEYS = {
    pygame.K_w: (0, 1),  # Move forward
    pygame.K_s: (0, -1),  # Move backward
    pygame.K_a: (1, 0),  # Move left
    pygame.K_d: (-1, 0)   # Move right
}

ROTATION_KEYS = {
    pygame.K_q: -1,  # Rotate left
    pygame.K_e: 1   # Rotate right
}

def get_action(keys):
    action_type = 0  # Default to translation
    translation_action = [0, 0]  # Default to no movement
    rotation_action = 0  # Default to no rotation

    for key, value in TRANSLATION_KEYS.items():
        if keys[key]:
            action_type = 0
            translation_action = list(value)
            break

    for key, value in ROTATION_KEYS.items():
        if keys[key]:
            action_type = 1
            rotation_action = value
            break

    # print(action_type)
    # print(translation_action)
    # print(rotation_action)

    #return np.array([action_type] + translation_action + [rotation_action], dtype=np.float32) # for translation and rotation
    return np.array(translation_action, dtype=np.float32) # translation only


# Initialize Pygame
pygame.init()

def start_env():
    try:
        env = gym.make('ContinuousMazeEnv-v1')
        print("Environment `ContinuousMazeEnv-v1` created successfully.")
        wrapped_env = FlattenObservation(env)
        print("Environment `ContinuousMazeEnv-v1` wrapped successfully.")
        observation, info = wrapped_env.reset(seed=None)
        wrapped_env.render()
        print("Environment reset successfully.")
        print(observation, info)
    except gym.error.Error as e:
        print(f"Failed to create or reset environment: {e}")
    return wrapped_env

wrapped_env = start_env()
done = False

# # Use this loop for continuous movement 
# while not done:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True
#             break


#     keys = pygame.key.get_pressed()
#     action = get_action(keys)
#     observation, reward, terminated, truncated, info = wrapped_env.step(action)
#     wrapped_env.render()

# Use this loop to pause at each time step (useful for simulating what how the agent moves when training)
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    action_taken = False
    while not action_taken:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                action = get_action(keys)
                #if any(action[1:3]) or action[3]:  # Check if any action is taken
                if any(action[0:2]):
                    action_taken = True
                    break


    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")

    if episode_rewards:
        episode_rewards.append(episode_rewards[-1] + reward)  # Cumulative reward
    else:
        episode_rewards.append(reward)  # Initial reward

    if terminated:
        all_rewards.append(episode_rewards)
        episode_rewards = []
        wrapped_env.close()
        wrapped_env = start_env()

    wrapped_env.render()

all_rewards.append(episode_rewards) #append rewards of unfinished episodes

env.close()
wrapped_env.close()
pygame.quit()

# Plot the cumulative rewards vs. timesteps for each episode
plt.figure()
for episode_index, episode_rewards in enumerate(all_rewards):
    plt.plot(episode_rewards, label=f'Episode {episode_index + 1}')
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward vs. Timesteps per Episode')
plt.legend()
plt.show()