import torch
import torch.nn as nn
import numpy as np
from rl.nets.q_network import QNetwork
from rl.buffers.replay_buffer import ReplayBuffer

# DQN (Deep Q-Network) - Mnih et al. 2015
# Q-Learning에 신경망을 결합한 알고리즘
#
# 기존 Q-Learning의 한계:
# 1. Q테이블: 상태가 연속적인 실수값이면 테이블 표현 불가
# 2. 매 스텝 바로 학습: 연속된 경험의 상관관계 → 학습 불안정
#
# DQN의 해결책:
# 1. Q테이블 → 신경망으로 대체: Q(s,a;θ) ≈ Q*(s,a)
# 2. Replay Buffer: 경험 (s,a,r,s',done) 저장 → 랜덤 샘플링으로 상관관계 제거
# 3. Target Network: TD target 계산용 별도 네트워크 → 학습 안정화
#
# Q-Learning 업데이트식:
# Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
#
# DQN loss:
# L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
# θ⁻: Target Network 파라미터 (고정), θ: Q Network 파라미터 (업데이트)
#
# TD target = r + γ max_a' Q(s',a';θ⁻)  ← Target Network로 계산 (고정된 목표)
# TD error  = TD target - Q(s,a;θ)       ← Q Network로 계산
#
# off-policy: Replay Buffer의 과거 경험으로 학습 → Q-Learning 선택 이유
# (SARSA 같은 on-policy는 현재 policy의 경험만 사용 가능)


class DQN:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_dim,
        lr,
        gamma,
        buffer_capacity,
        batch_size,
        target_update_freq,
        activation_func=nn.ReLU,
    ):
        self.n_actions = n_actions
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.gamma = gamma
        self.activation_func = activation_func

        self.C = 0

        # Q Network
        self.q_network = QNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            activation=self.activation_func,
        )
        # Target Network
        self.target_q_network = QNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            activation=self.activation_func,
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.batch_size = batch_size

    def select_action(self, state, eps):
        # ε-greedy로 action 선택
        # state를 텐서로 변환, 신경망은 배치 단위 입력(batch_size, obs_dim)을 받음.
        # 하지만 그냥 하나의 state는 배치 차원이 없음 (obs_dim,) -> unsqueeze(0)로 차원 추가
        # state = [1.2, 0.3, -0.1, 0.5]        # shape: (4,)
        # state.unsqueeze(0)                     # shape: (1, 4) -> 차원 추가 -> 신경망에 먹이기 OK
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        u = np.random.rand()

        if u < eps:
            return np.random.randint(self.n_actions)
        else:
            # q net에 널고 각 action Q 값 출력 -> .argmax() 가장 큰 Q 값의 인덱스 반환 -> item()으로 추출
            return self.q_network(state_tensor).argmax().item()  # -> action

    def select_greedy_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.q_network(state_tensor).argmax().item()

    def push(self, state, action, reward, next_state, done):
        # Replay Buffer에 경험 저장
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Buffer에서 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # Target Network로 target 계산
        # next_states 배치를 target net에 넣어 Q 값 출력, dim=1은 (batch_size, n_actions) 텐서에서 action 방향으로 max 찾기
        # ex)    action0  action1
        # 경험0  [  0.3,    0.8  ]   → dim=1 → 0.8
        # 경험1  [  0.5,    0.2  ]   → dim=1 → 0.5
        # 경험2  [  0.1,    0.9  ]   → dim=1 → 0.9
        # torch.max는 values (최대값(들)), indices (최대값의 인덱스) 를 반환
        with torch.no_grad():  # PyTorch는 모든 연산의 gradient를 추적(역전파 위해), target 계산은 고정된 목표라서 gradien 계산 필요 X(Target net은 업데이트 X)
            target = rewards + self.gamma * torch.max(
                self.target_q_network(next_states), dim=1
            ).values * (1 - dones)

        # Q Network로 현재 Q값 출력 (batch_size, n_actions)
        q = self.q_network(states)

        # 실제 선택한 action에 대한 Q 값 뽑기

        # gather(dim, index): index가 가리키는 위치의 값을 뽑아옴, dim = 1 -> (batch_size, n_actions)에서 n_actions 방향으로 인덱싱하기
        # gather는 입력 텐서(q)와 index 텐서의 차원 수가 같아야 하므로 actions.unsqueeze(1)
        # actions.unsqueeze(1) -> ex) actions = [1, 0, 1] (batch_size,) -> [[1], [0], [1]] (batch_size, 1)
        # [[1], [0], [1]] -> 경험 0에서는 action1([1]) 위치 뽑고 경험 1에서는 action0([0]) 위치 뽑고,,,
        # q.gather(1, actions.unsqueeze(1)) -> n_actions 방향의 열들(모든 action들) 중에 실제로 행했던 action들의 Q 값들을 뽑아옴
        # ex) squeeze(1)으로 [[0.8], [0.5], [0.9]]  →  [0.8, 0.5, 0.9]  # (batch_size,)
        q = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # loss 계산 및 역전파
        loss = nn.functional.mse_loss(q, target)

        self.optimizer.zero_grad()
        # optimizer에 등록된 파라미터 업데이트
        loss.backward()
        self.optimizer.step()

        self.C += 1

        # C 스텝마다 타겟 네트워크에 파라미터 복사
        if self.C >= self.target_update_freq:
            self.update_target()
            self.C = 0

    def update_target(self):
        # Target Network에 Q Network 파라미터 복사
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        return
