import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


# this is inspired by Dreamer v2

# ----------------------------------
# Encoder: low-res image -> embedding
# ----------------------------------
class LowResCNNEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 48, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        # x: [B, C=1, H=8, W=48]
        return self.conv(x)

# ----------------------------------
# Stochastic RSSM (DreamerV2-style)
# ----------------------------------
class RSSM(nn.Module):
    def __init__(self,
                 latent_dim=64,
                 stoch_vars=32,
                 stoch_classes=32,
                 hidden_size=600,
                 action_dim=3):
        super().__init__()
        self.stoch_vars = stoch_vars
        self.stoch_classes = stoch_classes
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        # Prior: GRU + MLP to produce logits
        self.rnn = nn.GRUCell(stoch_vars * stoch_classes + action_dim, hidden_size)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, stoch_vars * stoch_classes)
        )
        # Posterior: combine hidden state + encoder embed
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_size + latent_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, stoch_vars * stoch_classes)
        )

    def init_state(self, batch_size, device):
        # initialize hidden and stochastic state
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        # uniform one-hot prior across classes
        init_logits = torch.zeros(batch_size, self.stoch_vars, self.stoch_classes, device=device)
        z = OneHotCategorical(logits=init_logits).sample()
        z = z.view(batch_size, -1)
        return {'h': h, 'z': z}

    def forward(self, prev_state, action, embed=None):
        # prev_state: dict with 'h',[B,H] and 'z',[B,stoch_vars*stoch_classes]
        # action: [B,action_dim]; embed: [B,latent_dim] for posterior
        x = torch.cat([prev_state['z'], action], dim=-1)
        h = self.rnn(x, prev_state['h'])  # [B,hidden_size]
        # Prior logits
        prior_logits = self.prior_mlp(h).view(-1, self.stoch_vars, self.stoch_classes)
        prior_dist = OneHotCategorical(logits=prior_logits)
        prior_z = prior_dist.rsample().view(-1, self.stoch_vars * self.stoch_classes)
        # Posterior (only if embed provided)
        if embed is not None:
            post_input = torch.cat([h, embed], dim=-1)
            post_logits = self.post_mlp(post_input).view(-1, self.stoch_vars, self.stoch_classes)
            post_dist = OneHotCategorical(logits=post_logits)
            post_z = post_dist.rsample().view(-1, self.stoch_vars * self.stoch_classes)
        else:
            post_dist, post_z = None, prior_z
        return {'h': h, 'z': post_z}, prior_dist, post_dist

# ----------------------------------
# Decoders for observation & reward
# ----------------------------------
class ObservationDecoder(nn.Module):
    def __init__(self,
                 stoch_vars=32,
                 stoch_classes=32,
                 hidden_size=600,
                 output_channels=1):
        super().__init__()
        in_dim = stoch_vars * stoch_classes + hidden_size
        # MLP -> deconv
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64 * 8 * 48), nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 8, 48)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, state):
        # state: dict with 'h' and 'z'
        x = torch.cat([state['h'], state['z']], dim=-1)
        x = self.fc(x)
        return self.deconv(x)

class RewardDecoder(nn.Module):
    def __init__(self,
                 stoch_vars=32,
                 stoch_classes=32,
                 hidden_size=600):
        super().__init__()
        in_dim = stoch_vars * stoch_classes + hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state):
        x = torch.cat([state['h'], state['z']], dim=-1)
        return self.mlp(x)

# ----------------------------------
# Actor & Critic (latent-space)  
# ----------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=400, depth=4):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_size] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self,
                 stoch_vars=32,
                 stoch_classes=32,
                 hidden_size=600,
                 action_dim=3,
                 mlp_size=400,
                 mlp_depth=4):
        super().__init__()
        in_dim = stoch_vars * stoch_classes + hidden_size
        self.mlp = MLP(in_dim, action_dim, hidden_size=mlp_size, depth=mlp_depth)
    def forward(self, state):
        x = torch.cat([state['h'], state['z']], dim=-1)
        logits = self.mlp(x)
        return OneHotCategorical(logits=logits)

class Critic(nn.Module):
    def __init__(self,
                 stoch_vars=32,
                 stoch_classes=32,
                 hidden_size=600,
                 mlp_size=400,
                 mlp_depth=4):
        super().__init__()
        in_dim = stoch_vars * stoch_classes + hidden_size
        self.mlp = MLP(in_dim, 1, hidden_size=mlp_size, depth=mlp_depth)
    def forward(self, state):
        x = torch.cat([state['h'], state['z']], dim=-1)
        return self.mlp(x)

# ----------------------------------
# Helper: imagination rollout
# ----------------------------------
def imagine_ahead(rssm, actor, start_state, horizon=15):
    """
    Perform imagination rollout from start_state for `horizon` steps.
    Returns list of imagined states.
    """
    states, actions = [], []
    state = start_state
    for _ in range(horizon):
        dist = actor(state)
        action = dist.rsample()  # differentiable
        state, _, _ = rssm(state, action, embed=None)
        states.append(state)
        actions.append(action)
    return states, actions
