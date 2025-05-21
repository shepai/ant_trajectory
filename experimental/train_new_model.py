import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from newmodels import LowResCNNEncoder, RSSM, Actor, Critic
from grid_environment import CustomEnv

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment and models
env = CustomEnv()
encoder = LowResCNNEncoder(input_channels=1, latent_dim=64).to(device)
rssm = RSSM(latent_dim=64, action_dim=3).to(device)
actor = Actor(latent_dim=64, action_dim=3).to(device)
critic = Critic(latent_dim=64).to(device)

# Optimizers
optimizer_encoder_rssm = optim.Adam(list(encoder.parameters()) + list(rssm.parameters()), lr=1e-3)
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-3)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

# Tracking
episode_rewards = []
episode_lengths = []

def plot_metrics():
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(episode_lengths, label='Episode Length', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Preprocess state
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 1, 8, 48)
        latent = encoder(state)

        # Sample action from actor
        dist = actor(latent)
        action = dist.sample()
        action_onehot = F.one_hot(action, num_classes=3).float().to(device)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Preprocess next state
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        next_latent = encoder(next_state_tensor)

        # Predict next latent and reward
        predicted_next_latent, predicted_reward = rssm(latent, action_onehot)

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(device)

        # Compute losses
        target_value = reward_tensor + (1 - int(done)) * critic(next_latent).detach()
        target_value = target_value.view_as(critic(latent))

        loss_critic = nn.MSELoss()(critic(latent), target_value)
        loss_actor = -critic(latent).mean()

        loss_latent = nn.MSELoss()(predicted_next_latent, next_latent.detach())
        loss_reward = nn.MSELoss()(predicted_reward, reward_tensor)
        loss_rssm = loss_latent + loss_reward

        # Update critic
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # Update actor
        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

        # Update encoder and RSSM
        optimizer_encoder_rssm.zero_grad()
        loss_rssm.backward()
        optimizer_encoder_rssm.step()

        # Move to next state
        state = next_state
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")

    # Checkpoint and plot every 50 episodes
    if (episode + 1) % 50 == 0:
        torch.save({
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'encoder': encoder.state_dict(),
            'rssm': rssm.state_dict()
        }, f"checkpoint_ep{episode+1}.pt")
        plot_metrics()

print("Training completed.")
