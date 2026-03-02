import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        # 레이어 정의
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
