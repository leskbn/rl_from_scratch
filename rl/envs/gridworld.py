class GridWorld:
    def __init__(self, H=5, W=5, start=(0, 0), goal=(4, 4), max_steps=100):
        self.H = H
        self.W = W
        self.S = H * W
        self.start_id = self.encode(*start)
        self.goal_id = self.encode(*goal)
        self.current_id = self.start_id
        self.t = 0
        self.max_steps = max_steps
        self.n_actions = 4

    def reset(self, seed=None):
        """return state_id"""
        self.current_id = self.start_id
        self.t = 0
        return self.start_id

    def step(self, action: int):
        """return next_state_id, reward, terminated, truncated, info"""
        terminated = False
        truncated = False

        r, c = self.decode(self.current_id)

        if action == 0:  # Up
            nr, nc = r - 1, c
        elif action == 1:  # Right
            nr, nc = r, c + 1
        elif action == 2:  # Down
            nr, nc = r + 1, c
        elif action == 3:  # Left
            nr, nc = r, c - 1
        else:
            raise ValueError(f"Invalid action: {action} (expected 0..3)")

        nr = min(max(nr, 0), self.H - 1)
        nc = min(max(nc, 0), self.W - 1)

        next_state_id = self.encode(nr, nc)

        self.current_id = next_state_id

        reward = -1

        if next_state_id == self.goal_id:
            terminated = True

        info = {}
        self.t += 1

        if self.t >= self.max_steps:
            truncated = True

        return next_state_id, reward, terminated, truncated, info

    # DP용(핵심): step과 독립적인 순수 전이 함수
    def transition(self, s: int, a: int):
        """return s_next, reward, done"""
        done = False

        r, c = self.decode(s)

        if a == 0:  # Up
            nr, nc = r - 1, c
        elif a == 1:  # Right
            nr, nc = r, c + 1
        elif a == 2:  # Down
            nr, nc = r + 1, c
        elif a == 3:  # Left
            nr, nc = r, c - 1
        else:
            raise ValueError(f"Invalid action: {a} (expected 0..3)")

        nr = min(max(nr, 0), self.H - 1)
        nc = min(max(nc, 0), self.W - 1)

        s_next = self.encode(nr, nc)

        reward = -1

        if s_next == self.goal_id:
            done = True

        return s_next, reward, done

    # 유틸
    def encode(self, r: int, c: int) -> int:
        return r * self.W + c

    def decode(self, s: int):  # -> (r,c)
        return (s // self.W, s % self.W)
