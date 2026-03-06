import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=[64, 64], activation=nn.ReLU):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        in_dim = obs_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h

        layers.append(nn.Linear(in_dim, n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
