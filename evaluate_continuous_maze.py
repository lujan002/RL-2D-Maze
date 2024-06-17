import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import imageio
import time

import continuous_maze_env
env_id = 'ContinuousMazeEnv-v1'
vec_env = make_vec_env(env_id, n_envs=1)

# Load the trained model
start_time = time.time()
model = PPO.load("ppo_maze2")
print(f"Model loading time: {time.time() - start_time} seconds")

# Evaluate the model
start_time = time.time()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5, render=False)
print(f"Evaluation time: {time.time() - start_time} seconds")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Close the evaluation environment and create a new one with rendering for video recording
# vec_env.close()
# vec_env_norm.close()

# Reset the environment and start recording the video
start_time = time.time()
obs = vec_env.reset()
images = []
video_length = 5000

for _ in range(video_length):
    # if np.random.rand() < 0.1:
    #     action = np.array([vec_env_norm.action_space.sample()])qq
    # else:
    #     action, _ = model.predict(obs, state=None, deterministic=False)
    action, _ = model.predict(obs, state=None, deterministic=False)
    print(f"Action: {action}")
    obs, reward, done, info = vec_env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

    img = vec_env.envs[0].render()
    if img is not None and len(img.shape) == 3:  # Ensure img has correct dimensions
        images.append(img)
    else:
        print("Warning: Rendered image has unexpected shape")
    if done:
        obs = vec_env.reset()

print(f"Video recording time: {time.time() - start_time} seconds")

# Save the video
start_time = time.time()
if images:
    imageio.mimsave('maze_agent.mp4', [np.array(img) for img in images], fps=30)
    print(f"Video saving time: {time.time() - start_time} seconds")
else:
    print("Error: No images to save")

# Close the environment
vec_env.close()
