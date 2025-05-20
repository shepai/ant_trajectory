import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from trajectory_code.trajectory_process_functions import transform_model_trajects
from grid_environment import environment
from RL_code.DQNAgent import DQNAgent
import numpy as np
import datetime 
import os
def save_array_to_folder(base_dir, folder_name, fitness, pathways):
    folder_path = os.path.join(base_dir, folder_name)
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    # Save the array
    np.save(folder_path+"/fitnesses", fitness)
    
    transform_model_trajects(pathways, 
        image_path="/its/home/drs25/ant_trajectory/trajectory_code/testA_ant1_image.jpg", savefig=folder_path+"pathsTaken.pdf", x_scale=1)
    max_v = max(arr.shape[0] for arr in pathways)
    padded_pathways = []
    for arr in pathways:
        pad_length = max_v - arr.shape[0]
        if pad_length > 0:
            padded = np.vstack([arr, np.zeros((pad_length, 2))])
        else:
            padded = arr
        padded_pathways.append(padded)
    routes = np.stack(padded_pathways)
    np.save(folder_path+"/routes", routes)
    
    print(f"Array saved to: {folder_path}")

def run(experiment_name,epochs):
    env=environment() #call in demo environment
    image=env.getAntVision()
    agent=DQNAgent(image.shape,3,"cuda")

    T=1
    dt=0.01
    env.dt=dt
    path_hist=[]
    reward_hist=[]
    for j in range(epochs):
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
        reward_hist.append(total_reward)
        path_hist.append(np.array(env.trajectory).copy())
        print("Trial",j,"Reward:",total_reward)

    save_array_to_folder("/its/home/drs25/ant_trajectory/data/RL_TRIAL/", experiment_name, np.array(reward_hist), path_hist)

if __name__=="__main__":
    run("test",20)


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