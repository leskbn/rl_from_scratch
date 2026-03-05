import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=[64, 64], activation=nn.Tanh):
        super().__init__()
        # 단일 값이면 리스트로 변환
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        in_dim = obs_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.output(self.net(x))
