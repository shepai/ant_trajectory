import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from trajectory_code.trajectory_process_functions import transform_model_trajects
from grid_environment import environment
from RL_code.DQNAgent import DQNAgent
import numpy as np
import datetime 
env=environment() #call in demo environment
image=env.getAntVision()
agent=DQNAgent(image.shape,3,"cuda")

T=1
dt=0.01
env.dt=dt
path_hist=[]
for j in range(3000):
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
        if done: break
    path_hist.append(np.array(env.trajectory).copy())
    print("Trial",j,"Reward:",total_reward)

date=str(datetime.datetime.now()).replace(":","_")
##########
transform_model_trajects(path_hist, 
    image_path="/its/home/drs25/ant_trajectory/trajectory_code/testA_ant1_image.jpg", savefig="/its/home/drs25/ant_trajectory/data/RL_TRIAL/show"+date+".pdf", x_scale=1)
p=[]
for path in path_hist:
    if (len(np.arange(0,T,dt))-len(path)) > 0 :
        p.append(path+[[0,0]]*(len(np.arange(0,T,dt))-len(path)))
    else: p.append(path)
np.save("/its/home/drs25/ant_trajectory/data/RL_TRIAL/data"+str(date),np.array(p))

"""

##################
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

plt.show()"""