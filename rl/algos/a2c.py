import torch
import torch.nn as nn
import numpy as np
from rl.nets.policy_network import PolicyNetwork
from rl.nets.value_network import ValueNetwork

# A2C (Advantage Actor-Critic)
# TD Actor-Critic + 여러 환경 병렬 실행으로 학습 안정화
#
# TD Actor-Critic의 문제:
# - 환경 하나에서 나온 경험으로 업데이트 → 편향될 수 있음
# - gradient 분산이 높음
#
# A2C의 해결책: num_envs개 환경을 병렬로 돌려서 gradient 평균
# Var(평균) = Var(단일) / N → num_envs개 병렬이면 분산 N배 감소
#
# TD Actor-Critic과 비교:
# TD AC:  env 1개 → (obs_dim,) → 매 스텝 업데이트
# A2C:    env N개 → (num_envs, obs_dim) → 매 스텝 배치 업데이트
#
# A3C와 비교:
# A3C: 여러 워커가 비동기적으로 학습 → 멀티스레딩 필요, 구현 복잡
# A2C: 여러 환경을 동기적으로 학습 → 구현 단순, 성능 비슷
#
# 업데이트식:
# δ_t = r + γV(s') - V(s)  (TD error = Advantage)
#
# Actor loss  = -δ_t * log π(a|s)  → 평균(mean)으로 배치 처리
# Critic loss = δ_t²               → 평균(mean)으로 배치 처리


class A2C:
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

    def select_action(self, states):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(states, dtype=torch.float32)
        probs = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # tensor
        log_prob = dist.log_prob(action)

        return action, log_prob

    def select_greedy_action(self, states):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        action = probs.argmax().item()

        return action

    def update(self, states, next_states, rewards, log_probs, dones):
        # Advantage(TD error) 계산
        state_tensor = torch.tensor(states, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            # .squeeze(1)으로 value net 출력이 (num_envs, 1) 이기 때문에 (num_envs,)로 바꿔줌
            next_state_value = self.value_network(next_state_tensor).squeeze(1) * (
                1.0 - dones_tensor
            )

        td_err = (
            rewards_tensor
            + self.gamma * next_state_value
            - self.value_network(state_tensor).squeeze(1)
        )

        # loss 계산 및 역전파
        actor_loss = (-td_err * log_probs).mean()
        critic_loss = (td_err**2).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
