import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
import numpy as np
import datetime 
import os
from trajectory_code.trajectory_process_functions import transform_model_trajects
from grid_environment import environment
from RL_code.DQNAgent import DQNAgent
import torch
import datetime
import json
import csv
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


#just adding some extra saving of things
def run(experiment_name, epochs, save_dir="./trained_models"):
    env = environment()
    image = env.getAntVision()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(image.shape, 3, device)

    T = 1
    dt = 0.01
    env.dt = dt
    path_hist = []
    reward_hist = []
    epsilon_hist = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)
    csv_log_path = os.path.join(experiment_folder, "training_rewards.csv")

    # Save experiment config
    config = {
        "experiment_name": experiment_name,
        "epochs": epochs,
        "learning_rate": 1e-4,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon_start": 1.0,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "update_target_every": agent.update_target_every
    }
    with open(os.path.join(experiment_folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    with open(csv_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "TotalReward", "Epsilon"])

        for j in range(epochs):
            env.reset()
            state = env.getAntVision()
            total_reward = 0
            for t in np.arange(0, T, dt):
                action = agent.step(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                total_reward += reward
                if done:
                    break
            reward_hist.append(total_reward)
            epsilon_hist.append(agent.epsilon)
            path_hist.append(np.array(env.trajectory).copy())
            writer.writerow([j + 1, total_reward, agent.epsilon])
            print(f"Trial {j}: Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Save training results and model
    save_array_to_folder(experiment_folder, "trajectories", np.array(reward_hist), path_hist)
    model_path = os.path.join(experiment_folder, "model.pt")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot reward curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(reward_hist)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Plot epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_hist)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, "training_summary.png"))
    plt.show()

if __name__=="__main__":
    # run("test",20)
# I admit that this is not the most important change but I saw someone implementing this and I thought it looked super profesh, so might as well. If we don't like it, we can change it
    # this allows us to run the script from the terminal like "python RL_run.py --experiment baseline --epochs 50 --save_dir /tmp/my_training"
    run("TEST", 100, "/its/home/drs25/ant_trajectory/data/RL_TRIAL/")

    


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
