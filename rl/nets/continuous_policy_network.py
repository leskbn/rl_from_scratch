import torch.nn as nn
import torch


class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=[64, 64], activation=nn.Tanh):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std)
        return mean, std
