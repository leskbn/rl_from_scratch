import torch
import numpy as np
from rl.nets.actor_network import ActorNetwork
from rl.nets.critic_network import CriticNetwork
from rl.buffers.replay_buffer import ReplayBuffer


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

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
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
