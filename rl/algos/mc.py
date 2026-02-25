import numpy as np


# 에피소드 생성 유틸리티
# 주어진 policy로 환경과 상호작용하며 (state, reward) 시퀀스 수집
# DP와 달리 transition 모델 없이 실제 경험으로만 데이터 생성
def generate_episode(env, policy, max_steps=100):
    """return states: List[int], rewards: List[float]"""
    state = env.reset()
    t = 0
    states = []
    rewards = []

    while t < max_steps:
        a = np.random.choice(env.n_actions, p=policy[state])
        next_state, reward, terminated, truncated, _ = env.step(a)

        t += 1

        states.append(state)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    return states, rewards


# MC First-Visit Prediction
# 에피소드에서 각 상태 s를 처음 방문한 시점의 return G_t 만 사용해서 V(s) 추정
#
# G_t = r_{t+1} + γr_{t+2} + ... + γ^{T-t} r_T   (실제 return, bootstrap 없음)
# V(s) <- V(s) + (G_t - V(s)) / N(s)              (incremental mean)
#
# first-visit: 에피소드 내에서 s를 처음 방문한 t만 사용
# => 같은 에피소드에서 s를 여러 번 방문해도 첫 번째만 카운트
def mc_first_visit_prediction(env, policy, num_episodes, gamma, max_steps=100):
    """return V: shape (S,)"""
    V = np.zeros(env.S)
    N = np.zeros(env.S, dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, rewards = generate_episode(env, policy, max_steps)
        returns = np.zeros(len(states))

        # 뒤에서부터 return G_t 계산 (G_t = r_{t+1} + γ G_{t+1})
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        visited = set()

        for t in range(len(states)):
            G_t = returns[t]
            s = states[t]
            r, c = env.decode(s)
            if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                continue
            if s not in visited:
                visited.add(s)
                N[s] += 1
                # incremental mean: 모든 에피소드를 다 모은 뒤 평균 내는 대신
                # 에피소드마다 즉시 업데이트
                V[s] += (G_t - V[s]) / N[s]

    return V


# MC Every-Visit Prediction
# first-visit과 달리 에피소드 내에서 s를 방문할 때마다 모두 사용
# 방문 횟수가 많아질수록 더 빠르게 수렴하지만 bias가 생길 수 있음
#
# G_t = r_{t+1} + γr_{t+2} + ... + γ^{T-t} r_T
# V(s) <- V(s) + (G_t - V(s)) / N(s)   (모든 방문에 대해)
def mc_every_visit_prediction(env, policy, num_episodes, gamma, max_steps=100):
    """return V: shape (S,)"""
    V = np.zeros(env.S)
    N = np.zeros(env.S, dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, rewards = generate_episode(env, policy, max_steps)

        # 뒤에서부터 G 계산하면서 바로 업데이트 (every-visit이므로 visited 체크 불필요)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            s = states[t]
            r, c = env.decode(s)
            if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                continue
            N[s] += 1
            # incremental mean
            V[s] += (G - V[s]) / N[s]

    return V


def mc_prediction(env, policy, num_episodes, gamma, max_steps=100, first_visit=True):
    if first_visit:
        return mc_first_visit_prediction(env, policy, num_episodes, gamma, max_steps)
    else:
        return mc_every_visit_prediction(env, policy, num_episodes, gamma, max_steps)


# e-greedy action selection
# 확률 e로 random action (exploration), 1-e로 greedy action (exploitation)
# => Q가 업데이트될수록 greedy action이 점점 좋아짐 (암묵적 policy improvement)
def select_action_eps_greedy(Q, s, eps, n_actions):
    u = np.random.rand()

    if u < eps:
        return np.random.randint(n_actions)
    else:
        return int(np.argmax(Q[s]))


# 에피소드 생성 유틸리티 (control용)
# policy 대신 Q와 e를 받아서 e-greedy로 action 선택
def generate_episode_sar(env, Q, eps, max_steps=100):
    """return states, actions, rewards"""
    state = env.reset()
    t = 0
    states = []
    rewards = []
    actions = []

    while t < max_steps:
        a = select_action_eps_greedy(Q, state, eps, env.n_actions)
        next_state, reward, terminated, truncated, _ = env.step(a)

        t += 1

        states.append(state)
        rewards.append(reward)
        actions.append(a)

        state = next_state

        if terminated or truncated:
            break

    return states, actions, rewards


# MC On-Policy Control (e-greedy)
# Q(s,a)를 추정하면서 동시에 e-greedy policy를 개선
# 명시적인 policy 배열 없이 Q 자체가 policy를 내포
#
# Q(s,a) <- Q(s,a) + (G_t - Q(s,a)) / N(s,a)   (incremental mean)
#
# on-policy: 행동하는 policy(e-greedy)와 학습하는 policy가 동일
# => 탐색(e)과 활용(greedy)을 동시에 수행
#
# GPI (Generalized Policy Iteration):
#   에피소드마다 Q 업데이트 (evaluation) + e-greedy로 행동 (improvement) 반복
def mc_control_on_policy(env, num_episodes, gamma, eps, max_steps, first_visit=True):
    """return Q: shape (S,A)"""
    Q = np.zeros((env.S, env.n_actions))
    N = np.zeros((env.S, env.n_actions), dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, actions, rewards = generate_episode_sar(env, Q, eps, max_steps)

        if first_visit:
            # 뒤에서부터 return 계산 후 first-visit만 업데이트
            returns = np.zeros(len(states))
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                returns[t] = G

            visited = set()

            for t in range(len(states)):
                s = states[t]
                a = actions[t]
                G_t = returns[t]

                r, c = env.decode(s)
                if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                    continue

                if (s, a) not in visited:
                    visited.add((s, a))
                    N[s, a] += 1
                    Q[s, a] += (G_t - Q[s, a]) / N[s, a]
        else:
            # every-visit: 뒤에서부터 G 계산하면서 바로 업데이트
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                s = states[t]
                a = actions[t]

                r, c = env.decode(s)
                if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                    continue

                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

    return Q
