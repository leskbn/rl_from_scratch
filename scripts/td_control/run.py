# python -m scripts.td_control.run

import numpy as np

from rl.envs.gridworld import GridWorld
from rl.algos.td_control import sarsa, q_learning
from rl.utils.visualize import print_policy_arrows


def greedy_policy_from_Q(env, Q):
    policy = np.zeros((env.S, env.n_actions))
    for s in range(env.S):
        r, c = env.decode(s)
        if hasattr(env, "obstacles") and (r, c) in env.obstacles:
            continue
        if s == env.goal_id:
            continue
        policy[s, np.argmax(Q[s])] = 1.0
    return policy


def main():
    # -------------------------
    # Env
    # -------------------------
    obstacles = [(2, 1), (3, 4), (3, 1)]
    env = GridWorld(
        H=6, W=6, start=(0, 0), goal=(4, 3), obstacles=obstacles, max_steps=100
    )

    # -------------------------
    # Hyperparams
    # -------------------------
    np.random.seed(0)
    gamma = 0.99
    alpha = 0.05
    eps = 0.2
    num_episodes = 100000

    # -------------------------
    # SARSA
    # -------------------------
    sarsa_Q = sarsa(env, num_episodes, alpha, gamma, eps)
    sarsa_policy = greedy_policy_from_Q(env, sarsa_Q)
    print("===== SARSA Q (max) =====")
    print(np.round(sarsa_Q.max(axis=1).reshape(env.H, env.W), 2))
    print("\n===== SARSA Greedy Policy =====")
    print_policy_arrows(env, sarsa_policy)

    # -------------------------
    # Q-learning
    # -------------------------
    qlearning_Q = q_learning(env, num_episodes, alpha, gamma, eps)
    qlearning_policy = greedy_policy_from_Q(env, qlearning_Q)
    print("\n===== Q-Learning Q (max) =====")
    print(np.round(qlearning_Q.max(axis=1).reshape(env.H, env.W), 2))
    print("\n===== Q-Learning Greedy Policy =====")
    print_policy_arrows(env, qlearning_policy)


if __name__ == "__main__":
    main()
