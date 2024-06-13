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
vec_env = make_vec_env(env_id, n_envs=64)


# Load the YAML configuration file
with open('ppo_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Print the loaded configuration to verify
print("Loaded configuration:")
print(yaml.dump(config, default_flow_style=False))

# Ensure all necessary parameters are present
required_keys = ['policy', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range',
                'ent_coef', 'learning_rate', 'vf_coef', 'max_grad_norm', 'use_sde', 'sde_sample_freq',
                'target_kl', 'tensorboard_log', 'policy_kwargs', 'seed', 'device',
                '_init_setup_model']

missing_keys = [key for key in required_keys if key not in config['ppo']]
if missing_keys:
    raise ValueError(f"Missing required PPO parameters in config: {missing_keys}")




# Extract the PPO parameters from the configuration
ppo_params = config['ppo']

# Convert certain keys to None if they are the string "None"
for key in ['seed', 'target_kl', 'policy_kwargs']:
    if ppo_params[key] == "None":
        ppo_params[key] = None

# Create the PPO model with extracted parameters
model = PPO(**ppo_params, env=vec_env, verbose=1)

# Print device information
print(f"Model device: {model.device}")

# Define the callback with a frequency of x steps
reward_logging_callback = RewardLoggingCallback(check_freq=10000)

# Train the model
model.learn(total_timesteps=100000, callback=reward_logging_callback)  # Adjust the number of timesteps as needed

# Save the model
model.save("ppo_maze")

# Close the environment
env.close()


    