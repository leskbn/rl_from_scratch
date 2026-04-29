import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from rl.nets.critic_network import CriticNetwork
from rl.nets.value_network import ValueNetwork
from rl.buffers.replay_buffer import ReplayBuffer

# region
# Soft Actor-Critic (SAC)
# paper: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
# Haarnoja et al., 2018

# ============================================================
# 핵심 아이디어: Maximum Entropy RL
# ============================================================
# 기존 RL:  J(π) = E[Σ r(s,a)]
# SAC:      J(π) = E[Σ r(s,a) + α * H(π(·|s))]
#
# H(π(·|s)) = -E[log π(a|s)]  ← policy의 entropy
# α = temperature (entropy 중요도 조절)
#   → α 크면 탐험 많이, α 작으면 reward에 집중
#   → reward_scale = 5 는 effective α = 1/5 = 0.2 와 동일

# ============================================================
# Soft Bellman Equation
# ============================================================
# Soft Q:  Q(s,a) = r + γ * V(s')
# Soft V:  V(s)   = E[Q(s,a) - α * log π(a|s)]
#                        ↑Q값    ↑entropy 보정
#
# → V(s)는 Q값에서 entropy를 뺀 것
# → entropy 높은 action일수록 V값에 덜 반영

# ============================================================
# Stochastic Policy & Reparameterization
# ============================================================
# Actor: state → (μ, σ) → Gaussian 샘플링 → tanh → action
#
# reparameterization trick:
#   ε ~ N(0, 1)
#   u = μ + σ * ε       ← gradient 흐름
#   a = tanh(u)         ← -1~1로 제한
#
# tanh squashing 보정 (log prob):
#   log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
#   → 변수 변환 공식: p(a) = p(u) * |du/da|
#   → da/du = 1 - tanh²(u) 이므로 역수를 log에 더함

# ============================================================
# 네트워크 구성 (총 5개)
# ============================================================
# Actor:         state → (μ, σ) → 샘플링 → tanh → action
# Critic1/2:     (state, action) → Q값  (Twin Critic)
# Value:         state → V값
# Value_target:  soft update용  ← actor_target 불필요!

# ============================================================
# Loss 함수
# ============================================================
# Value loss:
#   J(V) = E[(V(s) - (min(Q1,Q2) - α*log π(a|s)))²]
#   → V가 soft value를 추정하도록
#
# Critic loss:
#   J(Q) = E[(Q(s,a) - (r*scale + γ*V_target(s')))²]
#   → soft Bellman equation으로 Q 학습
#
# Actor loss:
#   J(π) = E[α*log π(a|s) - min(Q1,Q2)]
#   → entropy 높이면서 Q값도 최대화
#   → reparameterization으로 gradient 흐름

# ============================================================
# 업데이트 순서 (매 스텝)
# ============================================================
# 1. Value  업데이트
# 2. Actor  업데이트  ← critic backward 전에
# 3. Critic 업데이트
# 4. Value_target soft update: ψ̄ ← τψ + (1-τ)ψ̄

# ============================================================
# TD3와의 차이
# ============================================================
# TD3: Deterministic policy + noise로 탐험
# SAC: Stochastic policy + entropy 최대화로 탐험 내재화
#   → seed간 variance 작음 (더 안정적)
#   → actor_target 불필요 (Value_target이 대신)
# endregion


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=[64, 64], activation=nn.ReLU):
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
        # state 마다 다른 std
        self.log_std_layer = nn.Linear(in_dim, n_actions)

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # reparameterization
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)

        # log prob with tanh correction
        log_prob = Normal(mean, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a, log_prob


class SAC:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_dim,
        lr_actor,
        lr_critic,
        gamma,
        tau,
        buffer_size,
        batch_size,
        lr_value,
        reward_scale,
        alpha,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau

        self.lr_value = lr_value
        self.reward_scale = reward_scale
        self.alpha = alpha

        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size, continuous=True)

        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dims=self.hidden_dim,
            activation=nn.ReLU,
        )

        self.critic1 = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic2 = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )

        self.critic1_target = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic2_target = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.value = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.value_target = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.value_target.load_state_dict(self.value.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optim = torch.optim.Adam(
            self.critic1.parameters(), lr=self.lr_critic
        )
        self.critic2_optim = torch.optim.Adam(
            self.critic2.parameters(), lr=self.lr_critic
        )
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.lr_value)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state)
        return action.squeeze(0).numpy()

    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            features = self.actor.net(state)
            mean = self.actor.mean_layer(features)
            action = torch.tanh(mean)
        return action.squeeze(0).numpy()

    def soft_update(self, network, target):
        for param, target_param in zip(network.parameters(), target.parameters()):
            target_param.data = (
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            new_actions_v, log_probs_v = self.actor(states)
            value_target = (
                torch.min(
                    self.critic1(states, new_actions_v),
                    self.critic2(states, new_actions_v),
                )
                - self.alpha * log_probs_v
            )

        value_loss = ((self.value(states) - value_target) ** 2).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        new_actions_a, log_probs_a = self.actor(states)

        actor_loss = (
            self.alpha * log_probs_a
            - torch.min(
                self.critic1(states, new_actions_a),
                self.critic2(states, new_actions_a),
            )
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        y = (
            rewards * self.reward_scale
            + self.gamma * self.value_target(next_states) * (1 - dones)
        ).detach()

        critic_loss = (
            (self.critic1(states, actions) - y) ** 2
            + (self.critic2(states, actions) - y) ** 2
        ).mean()

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        self.soft_update(self.value, self.value_target)
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
