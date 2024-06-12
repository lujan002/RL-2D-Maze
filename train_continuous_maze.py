import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import yaml

import continuous_maze_env

# Define the custom callback
class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        current_time = time.time()
        if self.n_calls % self.check_freq == 0:
            # Retrieve the reward from the model's buffer
            mean_reward = np.mean(self.model.rollout_buffer.rewards)
            elapsed_time = current_time - self.start_time
            print(f"Step {self.num_timesteps}: Mean Reward: {mean_reward}, Time Elapsed: {elapsed_time:.2f} seconds")
            elapsed_time = current_time - self.start_time
        return True
    
env_id = 'ContinuousMazeEnv-v1'
env = gym.make('ContinuousMazeEnv-v1', render_mode=None)
vec_env = make_vec_env(env_id, n_envs=12)

# Load the YAML configuration file
with open('ppo_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract the PPO parameters from the configuration
ppo_params = config['ppo']
model = PPO(**ppo_params, env=vec_env, verbose=1)

# Define the callback with a frequency of x steps
reward_logging_callback = RewardLoggingCallback(check_freq=500)

# Train the model
model.learn(total_timesteps=10000, callback=reward_logging_callback)  # Adjust the number of timesteps as needed

# Save the model
model.save("ppo_maze")

# Close the environment
env.close()
