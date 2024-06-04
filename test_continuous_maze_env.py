import gymnasium as gym
import continuous_maze_env  # Ensure this imports your environment file
import pygame
import time 
from Maze import Maze
from gymnasium.wrappers import FlattenObservation
import sys
sys.path.append('Users\20Jan\TU Graz Code\2D_maze_project')

# Define the control keys
KEY_MAPPING = {
    pygame.K_w: (0, 1, 0),
    pygame.K_s: (0, -1, 0),
    pygame.K_a: (1, 0, 0),
    pygame.K_d: (-1, 0, 0),
    pygame.K_q: (0, 0, -1),
    pygame.K_e: (0, 0, 1)
}

def get_action(keys):
    action = [0, 0, 0]
    for key, value in KEY_MAPPING.items():
        if keys[key]:
            action[0] += value[0]
            action[1] += value[1]
            action[2] += value[2]
    return action

# Initialize Pygame
pygame.init()

try:
    env = gym.make('ContinuousMazeEnv-v1')
    print("Environment `ContinuousMazeEnv-v1` created successfully.")
    wrapped_env = FlattenObservation(env)
    print("Environment `ContinuousMazeEnv-v1` wrapped successfully.")
    observation, info = wrapped_env.reset()
    wrapped_env.render()
    print("Environment reset successfully.")
    print(observation, info)
except gym.error.Error as e:
    print(f"Failed to create or reset environment: {e}")

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    keys = pygame.key.get_pressed()
    action = get_action(keys)
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    wrapped_env.render()
    time.sleep(0.01)  # Small delay to control the frame rate

env.close()
wrapped_env.close()
pygame.quit()


