import torch
import numpy as np
from rl.nets.actor_network import ActorNetwork
from rl.nets.critic_network import CriticNetwork
from rl.buffers.replay_buffer import ReplayBuffer

# region
# Twin Delayed Deep Deterministic Policy Gradient (TD3)
# paper: Addressing Function Approximation Error in Actor-Critic Methods
# Fujimoto et al., 2018

# ============================================================
# 배경: DDPG의 문제점
# ============================================================
# DDPG: Deterministic policy gradient
#   ∇J(θ) = E[∇_a Q(s,a) * ∇_θ μ(s)]
#   → Q값 과대추정 → 불안정한 학습
#   → hyperparameter에 매우 민감

# ============================================================
# Q값 과대추정이 발생하는 이유
# ============================================================
# Bellman equation:
#   y = r + γ * max_a Q(s', a)
#
# 문제 1: Function Approximation Error
#   신경망 Q(s,a)는 완벽하지 않음 → 추정 오차 존재
#   Q(s,a) = Q*(s,a) + ε  (ε = 추정 오차)
#
# 문제 2: max 연산이 오차를 증폭
#   max_a Q(s', a) = max_a [Q*(s', a) + ε]
#                 ≥ max_a Q*(s', a)
#   → 오차가 양수든 음수든 max를 취하면 항상 과대추정!
#   → 이 과대추정된 값이 target y로 사용
#   → y로 Q를 학습 → Q도 과대추정
#   → 반복될수록 오차 누적 → 발산

# ============================================================
# 핵심 아이디어 1: Clipped Double Q-learning
# ============================================================
# 기존 DDPG: y = r + γ * Q_target(s', Actor_target(s'))
# TD3:       y = r + γ * min(Q1_target, Q2_target)
#
# → Critic 2개 독립 학습
# → min으로 과대추정 방지
# → 과소추정이 과대추정보다 낫다:
#   과소추정된 action → policy가 피함 → 큰 문제 없음
#   과대추정된 action → policy가 선호 → 잘못된 방향으로 학습

# ============================================================
# 핵심 아이디어 2: Delayed Policy Update
# ============================================================
# Critic: 매 스텝 업데이트
# Actor:  policy_freq(=2) 스텝마다 업데이트
#
# → Critic이 어느 정도 수렴한 후 Actor 업데이트
# → 불안정한 Q값으로 Actor 업데이트하는 것 방지
# → Soft update도 Actor 업데이트할 때만

# ============================================================
# 핵심 아이디어 3: Target Policy Smoothing
# ============================================================
# noise = clip(N(0, σ), -c, c)
# a' = clip(Actor_target(s') + noise, -1, 1)
#
# → target action에 noise 추가
# → Critic이 특정 action에 과적합 방지
# → Q값을 action 주변에서 smooth하게 추정

# ============================================================
# 네트워크 구성 (총 6개)
# ============================================================
# Actor:            state → action (deterministic)
# Actor_target:     soft update용
# Critic1/2:        (state, action) → Q값
# Critic1/2_target: soft update용

# ============================================================
# Loss 함수
# ============================================================
# Critic loss:
#   y = r + γ * min(Q1_target(s', a'), Q2_target(s', a'))
#   a' = clip(Actor_target(s') + clip(N(0,σ), -c, c), -1, 1)
#   J(Q) = E[(Q1(s,a) - y)² + (Q2(s,a) - y)²]
#
# Actor loss:
#   J(π) = -E[Q1(s, Actor(s))]
#   → Q1만 사용 (관례, 어느 쪽이든 같은 방향)
#   → Deterministic Policy Gradient:
#     ∇J = ∇_a Q1(s,a) * ∇_θ Actor(s)

# ============================================================
# 업데이트 순서
# ============================================================
# 매 스텝:
#   1. Critic loss 계산 & 업데이트
#   2. policy_freq마다:
#      - Actor loss 계산 & 업데이트
#      - Soft update: θ̄ ← τθ + (1-τ)θ̄

# ============================================================
# DDPG와의 차이
# ============================================================
# DDPG: 단일 Critic, 매 스텝 Actor 업데이트, noise 없음
# TD3:  Twin Critic, Delayed update, Target Policy Smoothing
#   → Walker2d: DDPG ~1843 vs TD3 ~4682 (논문 기준)
# endregion


class TD3:
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
        policy_freq=2,
        noise_clip=0.5,
        noise_std=0.2,
        action_high=None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.tau = tau
        self.batch_size = batch_size
        self.policy_freq = policy_freq
        self.noise_clip = noise_clip
        self.noise_std = noise_std
        self.steps = 0

        # Actor: deterministic policy → action 직접 출력
        # Tanh로 -1~1 범위 제한, 실제 범위는 run에서 action_high 곱해서 변환
        self.actor = ActorNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.actor_target = ActorNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin Critic: Q(s,a) 추정 → state + action을 입력으로 받음
        self.critic1 = CriticNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.critic2 = CriticNetwork(self.obs_dim, self.n_actions, self.hidden_dim)

        self.critic1_target = CriticNetwork(
            self.obs_dim, self.n_actions, self.hidden_dim
        )
        self.critic2_target = CriticNetwork(
            self.obs_dim, self.n_actions, self.hidden_dim
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Actor, Critic optimizer 각자 따로 — 업데이트 순서가 다르기 때문
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.lr_critic
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=self.lr_critic
        )

        self.replay_buffer = ReplayBuffer(self.buffer_size, continuous=True)

    def select_action(self, state, noise=0.1):
        # 학습용: action + noise로 탐험
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        action = action.squeeze(0).numpy()  # 배치 차원 제거 및 tensor → numpy
        action = action + noise * np.random.randn(self.n_actions)  # 탐험용 noise 추가
        return np.clip(action, -1, 1)  # noise로 범위를 넘을 수 있으므로 clip

    def select_greedy_action(self, state):
        # 평가용: noise 없이 deterministic action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        action = action.squeeze(0).numpy()
        return np.clip(action, -1, 1)

    def soft_update(self, network, target):
        # θ_target = τ*θ + (1-τ)*θ_target
        # 매 스텝 조금씩 업데이트 → DQN 하드 복사보다 안정적
        for param, target_param in zip(network.parameters(), target.parameters()):
            target_param.data = (
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return
        self.steps += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            noise = torch.normal(mean=0, std=self.noise_std, size=actions.shape)
            target_actions = self.actor_target(next_states) + torch.clamp(
                noise, -self.noise_clip, self.noise_clip
            )
            target_actions = torch.clamp(target_actions, -1, 1)
            q1 = self.critic1_target(next_states, target_actions)
            q2 = self.critic2_target(next_states, target_actions)
            y = rewards.unsqueeze(1) + self.gamma * (
                torch.min(q1, q2) * (1 - dones.unsqueeze(1))
            )

        # Critic 업데이트: Q(s,a)가 y에 가까워지도록
        critic1_loss = ((y - self.critic1(states, actions)) ** 2).mean()
        critic2_loss = ((y - self.critic2(states, actions)) ** 2).mean()
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Actor 업데이트: Q(s, Actor(s)) 최대화, s에서 어떤 action 값이 가장 좋은지 찾아가도록
        if self.steps % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            # Actor 먼저 업데이트
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update: Actor, Critic 각자 target network 업데이트
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
