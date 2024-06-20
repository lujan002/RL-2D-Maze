import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import imageio
import time
import matplotlib.pyplot as plt
import continuous_maze_env


# Initialize lists to store rewards and timesteps
all_rewards = []
episode_rewards = []
cumulative_reward = 0

env_id = 'ContinuousMazeEnv-v1'
vec_env = make_vec_env(env_id, n_envs=1)

# Load the trained model
start_time = time.time()
model = PPO.load("ppo_maze")
print(f"Model loading time: {time.time() - start_time} seconds")

# Evaluate the model
start_time = time.time()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5, render=False)
print(f"Evaluation time: {time.time() - start_time} seconds")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Close the evaluation environment and create a new one with rendering for video recording
vec_env.close()
env = gym.make('ContinuousMazeEnv-v1', render_mode='rgb_array')

# Reset the environment and start recording the video
start_time = time.time()
obs, _ = env.reset()
images = []
video_length = 5000

for _ in range(video_length):
    # if np.random.rand() < 0.1:
    #     action = env.action_space.sample()
    # else:
    #     action, _ = model.predict(obs, state=None, deterministic=False)
    action, _ = model.predict(obs, state=None, deterministic=False)
    print(f"Action: {action}")
    obs, reward, terminated, truncated, _ = env.step(action)
    #print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
    if episode_rewards:
        episode_rewards.append(episode_rewards[-1] + reward)  # Cumulative reward
    else:
        episode_rewards.append(reward)  # Initial reward

    img = env.render()
    if img is not None and len(img.shape) == 3:  # Ensure img has correct dimensions
        images.append(img)
    else:
        print("Warning: Rendered image has unexpected shape")
    if terminated or truncated:
        all_rewards.append(episode_rewards)
        episode_rewards = []
        obs, _ = env.reset()

all_rewards.append(episode_rewards) #append rewards of unfinished episodes

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


# Plot the cumulative rewards vs. timesteps for each episode
plt.figure()
for episode_index, episode_rewards in enumerate(all_rewards):
    plt.plot(episode_rewards, label=f'Episode {episode_index + 1}')
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward vs. Timesteps per Episode')
plt.legend()
plt.show()