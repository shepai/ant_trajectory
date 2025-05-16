import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from grid_environment import environment
from RL_code.DQNAgent import DQNAgent
import numpy as np

env=environment() #call in demo environment
image=env.getAntVision()
agent=DQNAgent(image.shape,3,"cpu")

T=1
dt=0.05

for j in range(100):
    env.reset()
    state= env.getAntVision()
    total_reward=0
    for t in np.arange(0,T,dt):
        action = agent.step(state)  # pick action using policy
        next_state, reward, done, _ = env.step(action)  # env reacts
        agent.store_transition(state, action, reward, next_state, done)  # remember
        agent.train_step()  # learn from experience
        state = next_state
        total_reward+=reward
    print("Trial",j,"Reward:",total_reward)