import torch
import torch.nn as nn
import numpy as np
from rl.nets.policy_network import PolicyNetwork


# REINFORCE (Policy Gradient)
# Q값 같은 가치함수 없이 policy π(a|s;θ) 를 직접 학습
#
# 목표: 누적 보상의 기댓값 최대화
# J(θ) = E_π[G_t]
#
# 환경이 확률적이라 J(θ)를 직접 계산 불가 → 샘플(에피소드)로 근사
# 샘플로 근사하려면 gradient도 기댓값 형태여야 함
#
# Policy Gradient Theorem:
# ∇J(θ) = E_π[G_t ∇log π(a_t|s_t;θ)]
#
# log trick: ∇π = π * ∇log π
# → ∇π를 π * ∇log π로 바꿔서 기댓값 형태 복원 → 샘플로 근사 가능
#
# 업데이트식 (gradient ascent):
# θ ← θ + α * G_t * ∇log π(a_t|s_t;θ)
# PyTorch는 최소화 기반이라 loss = -G_t * log π(a_t|s_t;θ) 로 변환
#
# MC 방식: 에피소드 끝난 후 G_t 계산 → 업데이트 (online 불가)
# G_t가 크면 → a_t 확률 높임, G_t가 작으면 → a_t 확률 낮춤
#
# DQN과 비교:
# DQN:       Q값 학습 → argmax로 action 선택 → 매 스텝 업데이트 (online)
# REINFORCE: policy 직접 학습 → 확률 샘플링으로 action 선택 → 에피소드 끝나고 업데이트 (offline)
#
# 단점: 에피소드 하나로 업데이트 → 분산 높음 → 학습 불안정
# 해결책: baseline (b) 도입 → loss = -(G_t - b) * log π → Actor-Critic으로 발전


class REINFORCE:
    def __init__(
        self, obs_dim, n_actions, hidden_dim, lr, gamma, activation_func=nn.ReLU
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.activation_func = activation_func

        # Policy Network
        self.policy_network = PolicyNetwork(
            self.obs_dim,
            self.n_actions,
            self.hidden_dim,
            activation=self.activation_func,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def select_action(self, state):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def select_greedy_action(self, state):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        action = probs.argmax().item()

        return action

    def update(self, rewards, log_probs):
        # G_t 계산
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # loss 계산 및 역전파
        loss = 0
        for t in range(len(log_probs)):
            loss += -returns[t] * log_probs[t]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
