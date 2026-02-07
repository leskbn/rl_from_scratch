import numpy as np


# Iterative Policy Evaluation
# v_{k+1}(s) = Σ_{a∈A} π(a|s) · Σ_{s'∈S} P_{ss'}^a · [ r(s,a,s') + γ · v_k(s') ]
# = v_{k+1}(s) = Σ_a π(a|s) [ R_s^a + γ Σ_{s'} P_{ss'}^a v_k(s') ]
# (Deterministic GridWorld에서는 Σ_{s'} P_{ss'}^a v_k(s') = v_k(s_next) 로 축약됨. (다음 상태는 하나로 정해짐 => P=1))
# transition 확률은 원래는 존재하여 DP에 필요하고 알아야 계산 가능
# => v_{k+1}(s) = Σ_a π(a|s) [ R_s^a + γ v_k(s') ] => 이걸 모든 state에 대해서
def policy_evaluation(
    env,
    policy,
    gamma=0.99,
    theta=1e-8,
    print_each_iter=False,
    max_print_iters=20,
    decimals=2,
):
    """policy: shape (S,A), return V: shape (S,)"""
    S = env.S
    A = env.n_actions

    # 모든 상태에서 가치를 가지고 있음. 0으로 초기화
    V = np.zeros(S)

    iteration = 0
    while True:
        delta = 0

        for s in range(S):  # 모든 상태 s에 대해 V 업데이트
            r, c = env.decode(s)
            if (r, c) in env.obstacles:
                continue
            if s == env.goal_id:
                continue  # terminal 상태는 가치를 0으로

            v_old = V[s]
            v_new = 0
            # v_{k+1}(s) = Σ_{a∈A} π(a|s) · Σ_{s'∈S} P_{ss'}^a · [ r(s,a,s') + γ · v_k(s') ]
            for a in range(A):  # 한 state s에서 모든 action들에 대해 V[s] 업데이트
                p = policy[s, a]  # π(a|s)
                for prob, s_next, r, done in env.transition(s, a):
                    v_new += p * prob * (r + gamma * V[s_next] * (0.0 if done else 1.0))

            # 즉시즉시 비동기 업데이트. 다른 상태의 V 업데이트 시에 새로운 V를 바로 참조하게 됨
            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))

        if print_each_iter and iteration < max_print_iters:
            print(f"\n=== policy eval iter {iteration}, delta={delta:.3e} ===")
            print(np.round(V.reshape(env.H, env.W), decimals=decimals))

        iteration += 1
        if delta < theta:
            break

    return V


# Policy Improvement
# 현재 가치함수 V(= V^pi)를 보고, 각 상태 s에서 더 좋은 행동을 선택하도록
# 정책 π를 greedy하게 업데이트
def policy_improvement(env, V, gamma, policy):
    #   모든 상태에서 new_action(s)가 old_action(s)와 같으면 stable=True
    #   -> 정책이 더 이상 개선되지 않으므로 Policy Iteration 종료
    stable = True

    for s in range(env.S):
        r, c = env.decode(s)
        if (r, c) in env.obstacles:
            continue
        if s == env.goal_id:
            continue

        old_action = np.argmax(policy[s, :])

        best_action = None
        best_q = -np.inf

        for a in range(env.n_actions):
            q = 0.0
            for prob, s_next, r, done in env.transition(s, a):
                q += prob * (r + gamma * V[s_next] * (0.0 if done else 1.0))

            if q >= best_q:
                best_q = q
                best_action = a

        # 정책 갱신: s에서 best_action만 선택하도록
        policy[s, :] = 0
        policy[s, best_action] = 1

        if best_action != old_action:
            stable = False

    return policy, stable


# Policy Iteration
# 1) Policy Evaluation:
#   현재 정책 π를 고정하고, 그 정책의 가치함수 V^π를 수렴할 때까지 계산.
#   (Bellman expectation backup: Σ_a π(a|s)[r + γ V(s')])
#
# 2) Policy Improvement:
#   계산된 V^π를 이용해 각 상태에서 greedy 행동을 선택하도록 정책을 갱신.
#   π_new(s) = argmax_a [ r(s,a) + γ V^π(s') ]

# 명시적인 policy가 존재. evaluation과 improvement를 반복하여 정책을 최적으로


def policy_iteration(
    env,
    policy,
    gamma=0.99,
    theta=1e-8,
    print_each_iter=True,
    max_print_iters=20,
    decimals=2,
):
    it = 0
    while True:
        if print_each_iter:
            print(" ")
            print(
                f"*****************Policy iteration: " + str(it) + "*****************"
            )

        V = policy_evaluation(
            env, policy, gamma, theta, print_each_iter, max_print_iters, decimals
        )

        policy, stable = policy_improvement(env, V, gamma, policy)

        if stable:
            break
        it += 1

    return V, policy


# Value Iteration
# V_{k+1}(s) = max_a  Σ_{s'} P_{ss'}^a [ R_s^a + γ V_k(s') ]
# (Deterministic GridWorld에서는 Σ_{s'} ... 가 단일 next state로 축약됨)
# 각 상태 s에서 가능한 모든 행동 a를 시도해보고
# 그중 "가장 큰 가치(q)"를 만드는 행동의 값을 V(s)로 업데이트

# 명시적인 policy 존재 X, max를 이용해 가치함수를 최적으로 만들고 여기서 정책을 추출.


def value_iteration(
    env,
    gamma=0.99,
    theta=1e-8,
    print_each_iter=False,
    max_print_iters=20,
    decimals=2,
):
    """policy: shape (S,A), return V: shape (S,)"""
    S = env.S
    A = env.n_actions

    # 모든 상태에서 가치를 가지고 있음. 0으로 초기화
    V = np.zeros(S)

    iteration = 0
    while True:
        delta = 0

        for s in range(S):  # 모든 상태 s에 대해 V 업데이트
            max_q = -np.inf
            r, c = env.decode(s)
            if (r, c) in env.obstacles:
                continue
            if s == env.goal_id:
                continue  # terminal 상태는 가치를 0으로

            v_old = V[s]
            for a in range(A):  # 한 state s에서 모든 action들에 대해 V[s] 업데이트
                q = 0.0
                for prob, s_next, r, done in env.transition(s, a):
                    # q(s,a) = R_s^a + γ V_k(s_next)  (terminal로 끝나면 미래항은 0)
                    q += prob * (r + gamma * V[s_next] * (0.0 if done else 1.0))

                if q >= max_q:
                    max_q = q

            # V_{k+1}(s) = max_a q(s,a)
            v_new = max_q
            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))

        if print_each_iter and iteration < max_print_iters:
            print(f"\n====== iter {iteration}, delta={delta:.3e} ======")
            print(np.round(V.reshape(env.H, env.W), decimals=decimals))

        iteration += 1
        if delta < theta:
            break

    return V


def greedy_policy_from_V(env, V, gamma=0.99):

    policy = np.zeros((env.S, env.n_actions))

    for s in range(env.S):
        r, c = env.decode(s)
        if (r, c) in env.obstacles:
            continue
        if s == env.goal_id:
            continue
        best_action = None
        best_q = -np.inf

        for a in range(env.n_actions):
            q = 0.0
            for prob, s_next, r, done in env.transition(s, a):
                q += prob * (r + gamma * V[s_next] * (0.0 if done else 1.0))
            if q >= best_q:
                best_q = q
                best_action = a
        policy[s, best_action] = 1

    return policy
