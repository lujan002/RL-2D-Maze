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
vec_env = make_vec_env(env_id, n_envs=2)
vec_env_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Load the trained model
start_time = time.time()
model = PPO.load("ppo_maze")
print(f"Model loading time: {time.time() - start_time} seconds")

# Evaluate the model
start_time = time.time()
mean_reward, std_reward = evaluate_policy(model, vec_env_norm, n_eval_episodes=5, render=False)
print(f"Evaluation time: {time.time() - start_time} seconds")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Record a video
video_length = 500
images = []

# Close the evaluation environment and create a new one with rendering for video recording
# vec_env.close()
# vec_env_norm.close()

# Create a new vectorized environment and wrap it with VecNormalize
dummy_env = DummyVecEnv([lambda: gym.make(env_id, render_mode='rgb_array')])
vec_env_norm = VecNormalize(dummy_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Reset the environment and start recording the video
start_time = time.time()
obs = vec_env_norm.reset()
images = []
video_length = 500

for _ in range(video_length):
    action, _ = model.predict(obs, state=None, deterministic=False)
    print(f"Action: {action}")
    obs, reward, done, info = vec_env_norm.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

    img = vec_env_norm.envs[0].render()
    if img is not None and len(img.shape) == 3:  # Ensure img has correct dimensions
        images.append(img)
    else:
        print("Warning: Rendered image has unexpected shape")
    if done:
        obs = vec_env_norm.reset()

print(f"Video recording time: {time.time() - start_time} seconds")

# Save the video
start_time = time.time()
if images:
    imageio.mimsave('maze_agent.mp4', [np.array(img) for img in images], fps=30)
    print(f"Video saving time: {time.time() - start_time} seconds")
else:
    print("Error: No images to save")

# Close the environment
vec_env_norm.close()
