import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from newmodels import LowResCNNEncoder, RSSM, Actor, Critic
from grid_environment import environment

# Initialize environment and models
env = YourCustomEnv()
encoder = LowResCNNEncoder(input_channels=1, latent_dim=64)
rssm = RSSM(latent_dim=64, action_dim=3)
actor = Actor(latent_dim=64, action_dim=3)
critic = Critic(latent_dim=64)

# Optimizers
optimizer_encoder_rssm = optim.Adam(list(encoder.parameters()) + list(rssm.parameters()), lr=1e-3)
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-3)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Preprocess state
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 1, 8, 48)
        latent = encoder(state)

        # Sample action from actor
        dist = actor(latent)
        action = dist.sample()  # Scalar tensor
        action_onehot = F.one_hot(action, num_classes=3).float()

        # Step environment
        next_state, reward, done, _ = env.step(action.item())

        # Preprocess next state
        next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        next_latent = encoder(next_state)

        # Predict next latent and reward
        predicted_next_latent, predicted_reward = rssm(latent, action_onehot)

        # Compute losses
        target_value = reward + (1 - done) * critic(next_latent).detach()
        target_value = target_value.view_as(critic(latent))

        loss_critic = nn.MSELoss()(critic(latent), target_value)
        loss_actor = -critic(latent).mean()

        loss_latent = nn.MSELoss()(predicted_next_latent, next_latent.detach())
        loss_reward = nn.MSELoss()(predicted_reward, torch.tensor([[reward]], dtype=torch.float32))
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

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

print("Training completed.")
