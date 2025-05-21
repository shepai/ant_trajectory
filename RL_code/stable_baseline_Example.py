import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from grid_environment import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym

class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=128)

        # CNN expects (C, H, W), but observation is (H, W, C)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),  # C=1
            nn.ReLU(),
            nn.Flatten()
        )

        # Sample input to determine n_flatten
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()  # shape: (1, H, W, C)
            #sample = sample.permute(0, 3, 1, 2)  # shape: (1, C, H, W)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        #print("Input to CNN:", observations.shape)  # should be [batch, 1, 48, 8]
        return self.linear(self.cnn(observations))



# Optional: check your environment
env = CustomEnv()
check_env(env)

# Use a simple CNN policy
policy_kwargs = dict(
    features_extractor_class=SmallCNN,
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the model
model.learn(total_timesteps=10_000)

# Save it
model.save("ppo_custom_env")