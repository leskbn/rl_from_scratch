# python -m scripts.run_mc

import numpy as np

from rl.envs.gridworld import GridWorld
from rl.algos.mc import mc_prediction, mc_control_on_policy
from rl.utils.visualize import print_policy_arrows


def greedy_policy_from_Q(env, Q):
    policy = np.zeros((env.S, env.n_actions))
    for s in range(env.S):
        r, c = env.decode(s)
        if hasattr(env, "obstacles") and (r, c) in env.obstacles:
            continue
        if s == env.goal_id:
            continue
        a = int(np.argmax(Q[s]))
        policy[s, a] = 1.0
    return policy


def main():
    # -------------------------
    # Env
    # -------------------------
    obstacles = [(2, 1), (3, 4)]
    env = GridWorld(
        H=6, W=6, start=(0, 0), goal=(4, 3), obstacles=obstacles, max_steps=100
    )

    # -------------------------
    # Hyperparams
    # -------------------------
    np.random.seed(0)
    gamma = 0.99
    max_steps = 100
    num_episodes_pred = 10000
    num_episodes_ctrl = 200000
    eps = 0.2

    # -------------------------
    # MC Prediction (given policy)
    # -------------------------
    # 현재 policy를 통해 에피소드 샘플 생성 및 추정 V 계산
    # 예: uniform random policy
    policy = np.ones((env.S, env.n_actions)) / env.n_actions

    V = mc_prediction(
        env,
        policy,
        num_episodes=num_episodes_pred,
        gamma=gamma,
        max_steps=max_steps,
        first_visit=True,  # False: every-visit
    )

    print("===== MC Prediction (V) =====")
    print(np.round(V.reshape(env.H, env.W), 2))

    # -------------------------
    # MC Control (learn Q)
    # -------------------------
    # e-greedy policy로 에피소드 샘플 생성 및 Q 업데이트 -> 업데이트 된 Q로 다시 e-greedy (policy 개선) -> 반복...
    Q = mc_control_on_policy(
        env,
        num_episodes=num_episodes_ctrl,
        gamma=gamma,
        eps=eps,
        max_steps=max_steps,
        first_visit=True,  # False: every-visit control
    )

    V_from_Q = Q.max(axis=1)
    policy_greedy = greedy_policy_from_Q(env, Q)

    print("\n===== MC Control (V = max_a Q) =====")
    print(np.round(V_from_Q.reshape(env.H, env.W), 2))

    print("\n===== Greedy policy from Q =====")
    print_policy_arrows(env, policy_greedy)


if __name__ == "__main__":
    main()
