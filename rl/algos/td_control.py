import numpy as np


# ε-greedy action selection
# 확률 ε로 random action (exploration), 1-ε로 greedy action (exploitation)
# Q가 업데이트될수록 argmax Q[s]가 더 좋은 action을 가리킴 (암묵적 policy improvement)
def select_action_eps_greedy(Q, s, eps, n_actions):
    u = np.random.rand()

    if u < eps:
        return np.random.randint(n_actions)
    else:
        return int(np.argmax(Q[s]))


# SARSA (on-policy TD Control)
# Q(s,a)를 TD 방식으로 추정하면서 동시에 policy 개선
#
# Bellman equation:
# Q(s,a) = r + γ Q(s', a')
#
# TD target = r + γ Q(s', a')   (실제 reward + 현재 추정값 bootstrap),
#                                실제 상호작용을 통해 얻어낸 reward가 포함되어 좀 더 믿을만하므로 타겟으로
# TD error  = TD target - Q(s,a)
# Q(s,a) <- Q(s,a) + α * TD error -> α는 점진적으로 업데이트 시키기 위해. 없으면(α=1) 그냥 덮어씌워짐
#
# on-policy: 행동하는 policy(ε-greedy)와 학습하는 policy가 동일
# => 실제로 할 행동 a'로 업데이트 => 탐색 중 나쁜 행동도 Q에 반영 => 보수적
#
# (S, A, R, S', A') 를 한 묶음으로 사용해서 SARSA라는 이름
def sarsa(env, num_episodes, alpha, gamma, eps):
    """return Q: shape (S, A)"""
    Q = np.zeros((env.S, env.n_actions))

    for ep in range(num_episodes):
        s = env.reset()
        done = False

        # 에피소드 시작 전 첫 action 미리 뽑기 (S,A,R,S',A' 묶음을 위해)
        a = select_action_eps_greedy(Q, s, eps, env.n_actions)

        while not done:
            next_s, reward, terminated, truncated, _ = env.step(a)

            # 다음 action도 ε-greedy로 뽑기 (on-policy 핵심)
            next_a = select_action_eps_greedy(Q, next_s, eps, env.n_actions)

            # Q(s,a) <- Q(s,a) + α(r + γQ(s',a') - Q(s,a))
            Q[s, a] = Q[s, a] + alpha * (
                reward
                + gamma * Q[next_s, next_a] * (0.0 if terminated else 1.0)
                - Q[s, a]
            )

            s = next_s
            a = next_a

            if truncated or terminated:
                done = True

    return Q


# Q-Learning (off-policy TD Control)
# Q(s,a)를 TD 방식으로 추정하면서 동시에 최적 policy 개선
#
# Bellman optimality equation:
# Q*(s,a) = r + γ max_a' Q*(s', a')
#
# TD target = r + γ max_a' Q(s', a')   (항상 최적 행동 기준으로 bootstrap)
# TD error  = TD target - Q(s,a)
# Q(s,a) <- Q(s,a) + α * TD error
#
# off-policy: 행동하는 policy(ε-greedy)와 학습하는 policy(greedy)가 다름
# => 실제 행동과 무관하게 항상 max 기준으로 업데이트 => 공격적
# => SARSA보다 최적 policy에 빠르게 수렴
def q_learning(env, num_episodes, alpha, gamma, eps):
    """return Q: shape (S, A)"""
    Q = np.zeros((env.S, env.n_actions))

    for ep in range(num_episodes):
        s = env.reset()
        done = False

        while not done:
            a = select_action_eps_greedy(Q, s, eps, env.n_actions)
            next_s, reward, terminated, truncated, _ = env.step(a)

            # Q(s,a) <- Q(s,a) + α(r + γ max_a' Q(s',a') - Q(s,a))
            # max Q로 greedy policy가 선택할 action의 Q 값 방향으로
            Q[s, a] = Q[s, a] + alpha * (
                reward
                + gamma * np.max(Q[next_s]) * (0.0 if terminated else 1.0)
                - Q[s, a]
            )

            s = next_s

            if truncated or terminated:
                done = True

    return Q
