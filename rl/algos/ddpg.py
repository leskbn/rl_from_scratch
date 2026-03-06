import torch
import numpy as np
from rl.nets.actor_network import ActorNetwork
from rl.nets.critic_network import CriticNetwork
from rl.buffers.replay_buffer import ReplayBuffer

# DDPG (Deep Deterministic Policy Gradient) - Lillicrap et al. 2015
#
# DQN의 문제:
# - Q(s,a)에서 argmax로 action 선택 → discrete action만 가능
# - continuous action이면 무한한 action 중 max를 찾을 수 없음
#
# DDPG의 해결책: DQN + Actor-Critic
# - Actor: Q를 최대화하는 action을 직접 출력 (argmax 대체)
# - Critic: Q(s,a) 추정
#
# DQN에서 가져온 것:
# - Experience Replay: 경험 저장 후 랜덤 샘플링 → 경험 간 상관관계 제거
# - Target Network: 학습 안정화
#
# Actor-Critic에서 가져온 것:
# - Actor: policy network (deterministic)
# - Critic: Q-function
#
# PPO와 비교:
# - PPO (Stochastic):   state → (μ, σ) → Normal에서 샘플링 → action
# - DDPG (Deterministic): state → action 직접 출력 (샘플링 없음)
# - 같은 state면 항상 같은 action → 탐험을 위해 noise 필요
#
# Soft Update:
# - DQN: 주기적으로 target network 하드 복사
# - DDPG: 매 스텝 조금씩 업데이트 θ_target = τ*θ + (1-τ)*θ_target
# - 더 부드럽게 따라와서 학습 안정적
#
# Actor, Critic 각자 Target Network:
# - Critic target: y = r + γ * Q_target(s', Actor_target(s'))
# - Actor_target 없으면 target 계산할 때 Actor가 계속 바뀌어서 불안정
#
# Actor 업데이트:
# - actor_loss = -Q(s, Actor(s))
# - Critic을 통해 gradient가 Actor로 전달 → chain rule
# - Q값 최대화 = -Q값 최소화
# => Actor가 출력한 action을 Critic에 넣었을 때 Q값이 최대가 되도록 Actor를 업데이트
#
# Critic 업데이트:
# - y = r + γ * Q_target(s', Actor_target(s'))  ← bellman equation
# - critic_loss = (y - Q(s,a))²
# - DQN critic 업데이트랑 동일 — max Q(s',a')를 Actor_target으로 대체
#
# => Critic이 실제 경험을 바탕으로 업데이트하여 Q가 정확해지면
# => Actor도 action들에 대한 정확한 평가를 받아 더 좋은 action 방향으로 업데이트
#
# 탐험:
# - Deterministic policy라 샘플링 없음 → noise 추가
# - action = Actor(s) + noise
# - 학습 진행될수록 noise 줄여가는 방식도 있음 (noise decay)
#
# 전체 흐름:
# 1. Actor로 action 선택 + noise 추가
# 2. Replay Buffer에 경험 저장
# 3. 배치 샘플링
# 4. Critic 업데이트 (bellman equation)
# 5. Actor 업데이트 (Q 최대화)
# 6. Soft Update (Actor, Critic 각자)


class DDPG:
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

        # Actor: deterministic policy → action 직접 출력
        # Tanh로 -1~1 범위 제한, 실제 범위는 run에서 action_high 곱해서 변환
        self.actor = ActorNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.actor_target = ActorNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic: Q(s,a) 추정 → state + action을 입력으로 받음
        self.critic = CriticNetwork(self.obs_dim, self.n_actions, self.hidden_dim)
        self.critic_target = CriticNetwork(
            self.obs_dim, self.n_actions, self.hidden_dim
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Actor, Critic optimizer 각자 따로 — 업데이트 순서가 다르기 때문
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic
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

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            # Bellman equation: Q(s,a) = r + γ * max Q(s', a')
            # max Q(s', a'): s'에서 가장 좋은 action을 했을 때의 Q값
            #
            # 근데 continuous action이라 max를 못 찾음 -> 무한한 action 중 뭐가 max인지 모름
            # 그래서 Actor_target이 argmax 역할을 대신:
            # max Q(s', a') ≈ Q_target(s', Actor_target(s'))
            #
            # Actor_target은 Q를 최대화하는 action을 출력하도록 학습된 네트워크
            # -> continuous action에서 argmax 근사
            target_actions = self.actor_target(next_states)
            y = rewards.unsqueeze(1) + self.gamma * self.critic_target(
                next_states, target_actions
            ) * (1 - dones.unsqueeze(1))

        # Critic 업데이트: Q(s,a)가 y에 가까워지도록
        critic_loss = ((y - self.critic(states, actions)) ** 2).mean()

        # Actor 업데이트: Q(s, Actor(s)) 최대화, s에서 어떤 action 값이 가장 좋은지 찾아가도록
        # state들에 대해 현재 actor가 낸 action들의 Q를 재보고 업데이트.
        # gradient: -Q(s, Actor(s)) -> Critic 통해 Actor로 전달
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Actor 먼저 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward(
            retain_graph=True
        )  # Critic graph 유지 (Critic 업데이트에 재사용)
        self.actor_optimizer.step()

        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft Update: Actor, Critic 각자 target network 업데이트
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
