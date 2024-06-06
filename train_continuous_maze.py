import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import continuous_maze_env

# Define the custom callback
class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve the reward from the model's buffer
            mean_reward = np.mean(self.model.rollout_buffer.rewards)
            print(f"Step {self.num_timesteps}: Mean Reward: {mean_reward}")
        return True
    
env_id = 'ContinuousMazeEnv-v1'
env = gym.make('ContinuousMazeEnv-v1', render_mode=None)
vec_env = make_vec_env(env_id, n_envs=1)
model = PPO('MlpPolicy', vec_env, verbose=1)

# Define the callback with a frequency of x steps
reward_logging_callback = RewardLoggingCallback(check_freq=500)

# Train the model
model.learn(total_timesteps=1000, callback=reward_logging_callback)  # Adjust the number of timesteps as needed

# Save the model
model.save("ppo_maze")

# Close the environment
env.close()
