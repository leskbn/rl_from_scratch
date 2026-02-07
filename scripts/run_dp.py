# scripts/run_dp.py (skeleton)

import numpy as np
from rl.envs.gridworld import GridWorld
from rl.algos.dp import policy_evaluation
from rl.algos.dp import value_iteration
from rl.algos.dp import policy_iteration
from rl.algos.dp import greedy_policy_from_V
from rl.utils.visualize import print_policy_arrows


def main():
    # env 생성
    obstacles = [(2, 1), (3, 4)]
    env = GridWorld(
        H=6, W=6, start=(0, 0), goal=(4, 4), max_steps=100, obstacles=obstacles
    )

    # S, A 준비
    S = env.S
    A = env.n_actions

    gamma = 0.99

    # Policy_Evaluation
    # policy (예시: 모든 상태에서 각 행동을 택할 확률은 모두 동일)
    # policy = np.ones((S, A)) / A
    # V = policy_evaluation(env, policy, print_each_iter=True, max_print_iters=100)

    # Value Iteration
    V = value_iteration(env, gamma=gamma, print_each_iter=False, max_print_iters=100)
    policy_vi = greedy_policy_from_V(env, V, gamma=gamma)
    print("===== Value Iteration =====")
    print(np.round(V.reshape(env.H, env.W), decimals=2))
    print_policy_arrows(env, policy_vi)

    # Policy Iteration
    policy = np.ones((S, A)) / A
    V, policy = policy_iteration(env, policy, gamma=gamma, print_each_iter=False)
    print("===== Policy Iteration =====")
    print(np.round(V.reshape(env.H, env.W), decimals=2))
    print_policy_arrows(env, policy)


if __name__ == "__main__":
    main()
