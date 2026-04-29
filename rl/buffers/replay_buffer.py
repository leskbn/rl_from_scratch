import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, continuous=False):
        self.capacity = capacity
        self.continuous = continuous

        # 다음에 쓸 인덱스
        self.ptr = 0
        # 현재 저장 개수
        self.size = 0
        self.initialized = False

    def _init_arrays(self, state, action):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action)

        self.state_shape = state.shape
        self.action_shape = action.shape

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *self.state_shape), dtype=np.float32
        )

        if self.continuous:
            self.actions = np.zeros(
                (self.capacity, *self.action_shape), dtype=np.float32
            )
        else:
            self.actions = np.zeros((self.capacity,), dtype=np.int64)

        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.initialized = True

    def push(self, state, action, reward, next_state, done):
        if not self.initialized:
            # 첫 호출에 메모리 할당
            self._init_arrays(state, action)

        # ptr 위치에 transition 하나 저장
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)

        if self.continuous:
            self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        else:
            self.actions[self.ptr] = int(action)

        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # torch.from_numpy로 텐서 변환
        states = torch.from_numpy(self.states[idx])
        actions = torch.from_numpy(self.actions[idx])
        rewards = torch.from_numpy(self.rewards[idx])
        next_states = torch.from_numpy(self.next_states[idx])
        dones = torch.from_numpy(self.dones[idx])

        if self.continuous:
            actions = actions.float()
        else:
            actions = actions.long()

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):
        return self.size
