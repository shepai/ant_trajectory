#this runs the trained agent in th eenvironment (without learning)
#in terminal run it as python evaluate_agent.py --model ./trained_models/trial1_20240521_2105/model.pt --episodes 10 --render <-- model.pt is the weights, the trained_models folder comes from running RL_run.py


import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import csv

from RL_code.AntAgentCNN import AntAgentCNN
from grid_environment import environment

def load_model(model_path, input_channels, num_actions, device):
    model = AntAgentCNN(input_channels=input_channels, num_actions=num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# evaluation function
def evaluate_agent(model_path, episodes=10, render=False, save_dir="eval_results"):
    os.makedirs(save_dir, exist_ok=True)  # create output dir if missing

    env = environment()  #simulation environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_shape = env.getAntVision().shape  # shape of the visual input
    model = load_model(model_path, input_channels=input_shape[0], num_actions=3, device=device)

    rewards = []  # total reward per episode
    trajectories = []  # position list per episode

    # CSV log of reward per episode
    csv_log_path = os.path.join(save_dir, "reward_log.csv")
    with open(csv_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "TotalReward"])

        for episode in range(episodes):
            obs = env.reset()  # resets the agent position etc.
            state = env.getAntVision()  # get initial image
            done = False
            total_reward = 0
            path = []  # store agent positions

            while not done:
                # convert to tensor
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = model(state_tensor)
                    action = q_vals.argmax(dim=1).item()  # choose best action (greedy)

                # take step in environment
                _, reward, done, _ = env.step(action)
                state = env.getAntVision()
                total_reward += reward
                path.append(env.agent_pos.copy())

                if render:
                    env.render()

            rewards.append(total_reward)
            trajectories.append(np.array(path))
            writer.writerow([episode + 1, total_reward])
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}")

    # Save raw rewards array
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards))

    # Save padded trajectories (equal length)
    max_len = max(p.shape[0] for p in trajectories)
    padded = [np.vstack([p, np.zeros((max_len - p.shape[0], 2))]) for p in trajectories]
    np.save(os.path.join(save_dir, "trajectories.npy"), np.stack(padded))

    # Plot reward curve
    plt.plot(rewards)
    plt.title("Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(save_dir, "reward_plot.png"))
    plt.show()

    # Print summary stats
    print(f"Average reward: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--save_dir", type=str, default="eval_results", help="Directory to save results")
    args = parser.parse_args()

    evaluate_agent(args.model, episodes=args.episodes, render=args.render, save_dir=args.save_dir)
