import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from grid_environment import *
# Optional: check your environment
env = CustomEnv()
check_env(env)

# Use a simple CNN policy
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10_000)

# Save it
model.save("ppo_custom_env")