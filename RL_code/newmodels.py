#made a CNN encoder that will process the image of shape (8, 48, 1) into latent vector

class LowResCNNEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (32, 4, 24)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # (64, 2, 12)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # (128, 1, 6)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 1 * 6, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

#This predicts the next latent state and reward based on the current latent and action.
class RSSM(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.fc_reward = nn.Linear(latent_dim, 1)

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        next_latent = self.gru(x, latent)
        reward = self.fc_reward(next_latent)
        return next_latent, reward

#actor is the policy network, should output a probability distribution over the 3 discrete actions
class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, latent):
        logits = self.fc(latent)
        return torch.distributions.Categorical(logits=logits)

# critic is th evalue network , estimates the value of a latent state
class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, latent):
        return self.fc(latent)


