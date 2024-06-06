import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import time

import continuous_maze_env
env_id = 'ContinuousMazeEnv-v1'
env = make_vec_env(env_id, n_envs=1)

# Load the trained model
start_time = time.time()
model = PPO.load("ppo_maze")
print(f"Model loading time: {time.time() - start_time} seconds")

# Evaluate the model
start_time = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=False)
print(f"Evaluation time: {time.time() - start_time} seconds")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Record a video
video_length = 100
images = []

# Close the evaluation environment and create a new one with rendering for video recording
env.close()
env = gym.make(env_id, render_mode='rgb_array')  # This assumes the default render_mode is 'rgb_array' or supports rendering

start_time = time.time()
obs, info = env.reset()
for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _, _ = env.step(action)
    img = env.render()
    if img is not None and len(img.shape) == 3:  # Ensure img has correct dimensions
        images.append(img)
    else:
        print("Warning: Rendered image has unexpected shape")

print(f"Video recording time: {time.time() - start_time} seconds")

# Save the video
start_time = time.time()
if images:
    imageio.mimsave('maze_agent.mp4', [np.array(img) for img in images], fps=30)
    print(f"Video saving time: {time.time() - start_time} seconds")
else:
    print("Error: No images to save")

# Close the environment
env.close()
