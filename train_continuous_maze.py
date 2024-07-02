import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import time
import yaml

import continuous_maze_env




from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from stable_baselines3.common.torch_layers import MlpExtractor

class CustomMLPPolicy(ActorCriticPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomMLPPolicy, self).__init__(*args, **kwargs)
		self.net_arch = [
			dict(pi=[10, 10, 10, 4], vf=[10, 10, 10, 4])
		]

	def _build_mlp_extractor(self):
		"""
		Create the policy and value networks (multi-layer perceptrons).
		"""
		self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch,
										  activation_fn=nn.ReLU,
										  dropout_prob=0.1)

class MlpExtractor(nn.Module):
	def __init__(self, feature_dim, net_arch, activation_fn, dropout_prob):
		super(MlpExtractor, self).__init__()
		self.feature_dim = feature_dim
		self.activation_fn = activation_fn
		self.dropout_prob = dropout_prob

		# Build the policy network
		policy_layers = []
		last_layer_dim = feature_dim
		for layer in net_arch[0]['pi']:
			policy_layers.append(nn.Linear(last_layer_dim, layer))
			policy_layers.append(self.activation_fn())
			policy_layers.append(nn.Dropout(p=self.dropout_prob))
			last_layer_dim = layer
		self.policy_net = nn.Sequential(*policy_layers)

		# Build the value network
		value_layers = []
		last_layer_dim = feature_dim
		for layer in net_arch[0]['vf']:
			value_layers.append(nn.Linear(last_layer_dim, layer))
			value_layers.append(self.activation_fn())
			value_layers.append(nn.Dropout(p=self.dropout_prob))
			last_layer_dim = layer
		self.value_net = nn.Sequential(*value_layers)

	def forward(self, features):
		"""
		Forward pass in the policy and value networks.
		"""
		return self.policy_net(features), self.value_net(features)
      

# Define the custom callback
class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, var_threshold, min_steps, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.var_threshold = var_threshold
        self.min_steps = min_steps
        self.start_time = None
        self.rewards_buffer = []

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        current_time = time.time()

        # Append rewards to buffer
        self.rewards_buffer.append(np.mean(self.model.rollout_buffer.rewards))

        # Check variance every check_freq steps
        if self.n_calls % self.check_freq == 0 and len(self.rewards_buffer) > self.min_steps:
            # Calculate reward variance
            reward_variance = np.var(self.rewards_buffer[-self.min_steps:])
            
            elapsed_time = current_time - self.start_time
            print(f"Step {self.num_timesteps}: Reward Variance: {reward_variance}, Time Elapsed: {elapsed_time:.2f} seconds")

            # # Truncate episodes with low variance
            # if reward_variance < self.var_threshold:
            #     print(f"Terminating due to low reward variance: {reward_variance}")
            #     return False  # Terminate training early

        return True

env_id = 'ContinuousMazeEnv-v1'
env = gym.make('ContinuousMazeEnv-v1', render_mode=None)
vec_env = make_vec_env(env_id, n_envs=128)
#vec_env_norm = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=10.)

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
reward_logging_callback = RewardLoggingCallback(check_freq=20000//64, var_threshold=0.1, min_steps=100)

# Train the model
model.learn(total_timesteps=300000, callback=reward_logging_callback)  # Adjust the number of timesteps as needed

import sys
np.set_printoptions(threshold=sys.maxsize)

# Verify actions from model during training
# obs = vec_env.reset()
# for _ in range(10):
# 	action, _ = model.predict(obs)
# 	print(f"Predicted Action during evaluation: {action}")
# 	obs, reward, done, info = vec_env.step(action)
# 	# Reset only those environments that are done
# 	obs = np.array([vec_env.reset()[i] if done[i] else obs[i] for i in range(len(done))])
          
#do this only for the first vec env
# original_obs = vec_env_norm.get_original_obs()[0:4]
# print(f"Sample of original obs: {original_obs}")
# norm_obs = vec_env_norm.normalize_obs(original_obs)
# print(f"Sample of normalized obs: {norm_obs}")

# original_rew = vec_env_norm.get_original_reward()[0:4]
# print(f"Sample of original reward: {original_rew}")
# norm_rew = vec_env_norm.normalize_reward(original_rew)
# print(f"Sample of normalized reward: {norm_rew}")

# Save the model
try:
    model.save("ppo_maze")
except Exception as e:
    print(f"Could not save model: {e}")

# Close the environment
env.close()


    