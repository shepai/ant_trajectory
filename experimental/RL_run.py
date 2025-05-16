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
path_hist=[]
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
    path_hist.append(np.array(env.trajectory).copy())
    print("Trial",j,"Reward:",total_reward)

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
matplotlib.use('TkAgg')
num_paths = len(path_hist)
colors = plt.cm.viridis(np.linspace(0, 1, num_paths))
lines = []
for path in path_hist:
    segments = np.array([path[:-1], path[1:]]).transpose(1, 0, 2)
    lines.extend(segments)
# Flatten all paths into a LineCollection
line_collection = LineCollection(lines, cmap='viridis', norm=plt.Normalize(0, num_paths))
line_collection.set_array(np.repeat(np.arange(num_paths), path_hist[0].shape[0] - 1))
fig, ax = plt.subplots()
ax.add_collection(line_collection)
ax.autoscale()
cbar = plt.colorbar(line_collection, ax=ax)
cbar.set_label('Path index')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Paths with color bar')


#save everything 

plt.show()