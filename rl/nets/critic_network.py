import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=[64, 64], activation=nn.ReLU):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        in_dim = obs_dim + action_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # state + action 합치기
        return self.output(self.net(x))
