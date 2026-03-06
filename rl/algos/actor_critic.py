import torch
import torch.nn as nn
import numpy as np
from rl.nets.policy_network import PolicyNetwork
from rl.nets.value_network import ValueNetwork

# Actor-Critic (Vanilla)
# REINFORCE의 high variance 문제를 Critic(가치함수)으로 해결
#
# REINFORCE의 문제:
# θ ← θ + α * G_t * ∇log π(a_t|s_t;θ)
# G_t의 분산이 너무 높음 → 학습 불안정
#
# 해결책: baseline으로 V(s_t)를 빼서 Advantage 사용
# A_t = G_t - V(s_t)  ("예상보다 얼마나 좋았나")
# - A_t > 0: 예상보다 좋았어 → 이 action 확률 높임
# - A_t < 0: 예상보다 나빴어 → 이 action 확률 낮춤
# - A_t = 0: 예상대로 → 업데이트 없음
#
# V(s_t)를 추정하는 신경망이 Critic, policy를 학습하는 신경망이 Actor
#
# Actor loss  = -A_t * log π(a_t|s_t;θ)
# Critic loss = (G_t - V(s_t))²
#
# MC 방식: 에피소드 끝나고 G_t 계산 → 업데이트 (offline)
# → 에피소드 길수록 G_t 분산 커져서 LunarLander 같은 환경에 약함
#
# REINFORCE와 비교:
# REINFORCE: G_t로 업데이트, Value Network 없음
# Actor-Critic: A_t = G_t - V(s_t)로 업데이트, Critic이 V(s) 추정
#
# TD Actor-Critic과 비교:
# Vanilla AC: G_t 사용 (MC), 에피소드 끝나고 업데이트
# TD AC:      δ_t = r + γV(s') - V(s) 사용, 매 스텝 업데이트


class ActorCritic:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_dim,
        lr,
        gamma,
        policy_activation_func=nn.ReLU,
        value_activation_func=nn.Tanh,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.policy_activation_func = policy_activation_func
        self.value_activation_func = value_activation_func

        # Policy Network (Actor)
        self.policy_network = PolicyNetwork(
            self.obs_dim,
            self.n_actions,
            self.hidden_dim,
            activation=self.policy_activation_func,
        )

        # Value Network (Critic)
        self.value_network = ValueNetwork(
            self.obs_dim, self.hidden_dim, activation=self.value_activation_func
        )

        # Optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.lr
        )

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

    def update(self, states, rewards, log_probs):
        # Advantage, return 계산
        A = 0
        G = 0
        returns = []
        advantages = []

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        for t in range(len(states)):
            # value_network 출력이 (1, 1) shape이라서 squeeze
            state_tensor = torch.tensor(states[t], dtype=torch.float32).unsqueeze(0)
            V = self.value_network(state_tensor).squeeze()
            A = returns[t] - V
            advantages.append(A)

        # loss 계산 및 역전파
        actor_loss = 0
        critic_loss = 0
        for t in range(len(log_probs)):
            actor_loss += -advantages[t] * log_probs[t]
            critic_loss += advantages[t] ** 2

        self.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
