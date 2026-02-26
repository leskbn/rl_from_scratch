import torch
import torch.nn as nn


# GridWorld: 상태 = 격자 번호(정수) → Q테이블로 표현 가능
# CartPole: 상태 = 실수 벡터 → Q테이블 불가능 → 신경망 필요


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        # 레이어 정의
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        # forward pass
        return self.net(x)
