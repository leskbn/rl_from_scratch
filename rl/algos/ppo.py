import torch
import torch.nn as nn
import numpy as np
from rl.nets.policy_network import PolicyNetwork
from rl.nets.value_network import ValueNetwork

# region
# PPO (Proximal Policy Optimization) - Schulman et al. 2017
# TRPO의 아이디어를 단순화한 알고리즘
#
# TRPO의 문제:
# - KL divergence 제약 조건 → Conjugate Gradient + Line Search 필요
# - 구현 복잡, 계산 비쌈
#
# PPO의 해결책: KL 제약 대신 Clipping으로 단순화
# policy 변화량을 [1-ε, 1+ε] 범위로 강제로 제한
#
# Importance Sampling Ratio:
# r_t(θ) = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
# - r_t > 1: 새 policy가 이 action을 더 선호
# - r_t < 1: 새 policy가 이 action을 덜 선호
# - r_t = 1: 두 policy가 동일
#
# Clipped Surrogate Objective:
# L_CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
# - A_t > 0 (좋은 행동): r_t가 너무 커지면 clip → 너무 많이 확률 높이는 걸 막음
# - A_t < 0 (나쁜 행동): r_t가 너무 작아지면 clip → 너무 많이 확률 낮추는 걸 막음
# - min: 양방향으로 policy 변화 제한
#
# Total Loss:
# L = L_CLIP + c1 * L_VF - c2 * H
# - L_CLIP: Actor loss (clipped surrogate objective)
# - L_VF = (G_t - V(s))²: Critic loss
# - H = -Σ π log π: Entropy bonus (탐험 장려, policy가 너무 확정적이 되는 걸 방지)
#
# GAE (Generalized Advantage Estimation):
# A_t^GAE = Σ (γλ)^k δ_{t+k}  where δ_t = r_t + γV(s_{t+1}) - V(s_t)
# λ=0: TD error만 (low variance, high bias)
# λ=1: MC return (high variance, low bias)
# λ=0.95: 절충 (실용적으로 제일 많이 쓰임)
#
# 전체 흐름:
# 1. π_old로 T 스텝 경험 수집 (rollout)
# 2. GAE로 Advantage 계산
# 3. K번 반복 업데이트 (미니배치 샘플링)
# 4. π_new → π_old 복사
#
# A2C와 비교:
# A2C: 매 스텝 업데이트, 경험 한 번만 사용
# PPO: T 스텝 모아서 K번 반복 업데이트 → 데이터 효율적, 안정적
#
# discrete/continuous action space 둘 다 지원:
# discrete:   Categorical 분포, Softmax 출력
# continuous: Normal 분포, Gaussian (μ, σ) 출력


# PPO에서 T (rollout length)를 사용하는 이유:
# 에피소드 단위의 문제:
# - 에피소드 길이가 들쭉날쭉 → 배치 크기 불규칙 → 학습 불안정
#
# 매 스텝 업데이트의 문제:
# - 경험을 한 번만 쓰고 버림 → 비효율적
# - GAE 계산하려면 여러 스텝의 경험이 필요
#
# T의 장점:
# - 항상 고정된 크기의 배치 → 학습 안정적
# - 에피소드 경계에 상관없이 일정하게 업데이트
# - T / minibatch_size개의 미니배치가 항상 일정하게 나옴
# - K번 반복 업데이트할 때 배치 크기 보장
#
# T = 2048이면:
# - 2048 스텝 수집 → GAE 계산 → 64개 미니배치로 나눠서 K번 반복
# - 에피소드가 중간에 끊겨도 last_value로 bootstrap해서 보정

# PPO의 학습 파이프라인은 다른 알고리즘이 사용 가능
# r_t = π_new / π_old 로 policy 변화량을 측정하고 clip으로 제한하는 것 — PPO의 본질

# endregion


class PPO:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_dim,
        activation_func,
        lr,
        gamma,
        eps_clip,
        K_epochs,
        lam,
        c1,
        c2,
        minibatch_size,
        continuous,
        action_low=None,
        action_high=None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.activation_func = activation_func
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lam = lam

        self.c1 = c1
        self.c2 = c2

        self.minibatch_size = minibatch_size

        self.continuous = continuous

        # Policy Network (Actor)
        if self.continuous:

            from rl.nets.continuous_policy_network import ContinuousPolicyNetwork

            self.policy_network = ContinuousPolicyNetwork(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dims=self.hidden_dim,
                activation=self.activation_func,
            )

            self.policy_network_old = ContinuousPolicyNetwork(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dims=self.hidden_dim,
                activation=self.activation_func,
            )

        else:
            self.policy_network = PolicyNetwork(obs_dim, n_actions, hidden_dim)
            self.policy_network_old = PolicyNetwork(obs_dim, n_actions, hidden_dim)

        self.policy_network_old.load_state_dict(self.policy_network.state_dict())

        # Value Network (Critic)
        self.value_network = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation_func,
        )

        # Optimizer (하나로 합쳐도 됨)
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters())
            + list(self.value_network.parameters()),
            lr=self.lr,
        )

    def select_action(self, states):
        # states shape: (num_envs, obs_dim) 또는 (obs_dim,) 둘 다 처리
        state_tensor = torch.tensor(states, dtype=torch.float32)
        is_single = state_tensor.dim() == 1
        if is_single:
            state_tensor = state_tensor.unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            if self.continuous:
                mean, std = self.policy_network_old(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                probs = self.policy_network_old(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            value = self.value_network(state_tensor).squeeze(-1)

        if is_single:
            # 단일 state → 스칼라로 반환
            if self.continuous:
                return action.squeeze(0).numpy(), log_prob.item(), value.item()
            else:
                return action.item(), log_prob.item(), value.item()
        else:
            # 배치 → numpy array로 반환
            return action.numpy(), log_prob.numpy(), value.numpy()

    def select_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.continuous:
                mean, _ = self.policy_network(state_tensor)
                return mean.squeeze(0).numpy()  # mean이 greedy action
            else:
                probs = self.policy_network(state_tensor)
                return probs.argmax().item()

    def compute_gae(self, rewards, values, dones):
        # GAE (Generalized Advantage Estimation)
        # δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
        # A_t^GAE = Σ (γλ)^k δ_{t+k}  (뒤에서부터 계산)
        # G_t = A_t^GAE + V(s_t)  (Critic 학습용 return)
        # values[T] = last_value (에피소드 중간에 잘린 경우 bootstrap)
        gae = 0
        advantages = []
        returns = []

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * (0 if dones[t] else gae)
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        T = len(states)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(
            actions, dtype=torch.float32 if self.continuous else torch.long
        )
        log_probs_old_tensor = torch.tensor(log_probs_old, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        # 정규화: 평균 0, 분산 1로 만들어서 학습 안정화
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # K번 반복 — 같은 경험을 K번 재사용 (데이터 효율성)
        for t in range(self.K_epochs):
            # 매 epoch마다 랜덤 셔플 → 다양한 미니배치 구성
            indices = torch.randperm(
                T
            )  # 0부터 T-1까지 숫자를 랜덤하게 섞은 인덱스를 반환
            for start in range(0, T, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[
                    start:end
                ]  # 랜덤하게 만든 인덱스중에 64개씩 뽑아쓰기

                # 미니배치 추출
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_log_probs_old = log_probs_old_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # r_t = π_new / π_old 계산
                if self.continuous:
                    mean, std = self.policy_network(mb_states)
                    dist = torch.distributions.Normal(mean, std)
                    log_probs_new = dist.log_prob(mb_actions).sum(dim=-1)
                else:
                    probs = self.policy_network(mb_states)
                    dist = torch.distributions.Categorical(probs)
                    log_probs_new = dist.log_prob(mb_actions)

                r_t = torch.exp(log_probs_new - mb_log_probs_old)

                # Clipped loss 계산
                r_t_clipped = torch.clamp(r_t, 1 - self.eps_clip, 1 + self.eps_clip)
                # PyTorch는 최소화 기반이기 때문에 - 붙임
                actor_loss = -torch.min(
                    r_t * mb_advantages,
                    r_t_clipped * mb_advantages,
                ).mean()

                # Critic loss 계산
                critic_loss = (
                    (mb_returns - self.value_network(mb_states).squeeze(1)) ** 2
                ).mean()

                # Entropy bonus 계산
                if self.continuous:
                    entrophy_bonus = dist.entropy().sum(dim=-1).mean()
                else:
                    entrophy_bonus = dist.entropy().mean()

                # 합산해서 역전파
                # L = L_CLIP + c1 * L_VF - c2 * H
                total_loss = (
                    actor_loss + self.c1 * critic_loss - self.c2 * entrophy_bonus
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping: 업데이트가 너무 커지는 걸 방지
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters())
                    + list(self.value_network.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

        # K번 업데이트 끝나면 π_new → π_old 복사
        self.policy_network_old.load_state_dict(self.policy_network.state_dict())

        return
