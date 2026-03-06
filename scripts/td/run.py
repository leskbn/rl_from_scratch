# python -m scripts.td.run

import numpy as np

from rl.envs.gridworld import GridWorld
from rl.algos.td import td0, td_lambda_forward, td_lambda_backward

from rl.algos.dp import greedy_policy_from_V
from rl.utils.visualize import print_policy_arrows


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
    alpha = 0.05
    lam = 0.8
    num_episodes = 10000

    # uniform random policy
    policy = np.ones((env.S, env.n_actions)) / env.n_actions

    # -------------------------
    # TD(0) Prediction
    # -------------------------
    V_td0 = td0(env, policy, num_episodes=num_episodes, alpha=alpha, gamma=gamma)

    print("===== TD(0) Prediction (V) =====")
    print(np.round(V_td0.reshape(env.H, env.W), 2))
    policy_td0 = greedy_policy_from_V(env, V_td0, gamma=gamma)
    print("\n===== TD(0) Greedy Policy =====")
    print_policy_arrows(env, policy_td0)

    # -------------------------
    # TD(λ) Forward View
    # -------------------------
    V_fwd = td_lambda_forward(
        env, policy, num_episodes=num_episodes, alpha=alpha, gamma=gamma, lam=lam
    )

    print(f"\n===== TD(λ={lam}) Forward View (V) =====")
    print(np.round(V_fwd.reshape(env.H, env.W), 2))

    policy_fwd = greedy_policy_from_V(env, V_fwd, gamma=gamma)
    print(f"\n===== TD(λ={lam}) Forward Greedy Policy =====")
    print_policy_arrows(env, policy_fwd)

    # -------------------------
    # TD(λ) Backward View
    # -------------------------
    V_bwd = td_lambda_backward(
        env, policy, num_episodes=num_episodes, alpha=alpha, gamma=gamma, lam=lam
    )

    print(f"\n===== TD(λ={lam}) Backward View (V) =====")
    print(np.round(V_bwd.reshape(env.H, env.W), 2))

    policy_bwd = greedy_policy_from_V(env, V_bwd, gamma=gamma)
    print(f"\n===== TD(λ={lam}) Backward Greedy Policy =====")
    print_policy_arrows(env, policy_bwd)

    # -------------------------
    # 비교: 차이
    # -------------------------
    print(f"\n===== Forward vs Backward 차이 (max|diff|) =====")
    print(f"{np.max(np.abs(V_fwd - V_bwd)):.4f}  (거의 0에 가까울수록 좋음)")

    print(f"\n===== TD(0) vs Forward 차이 (max|diff|) =====")
    print(f"{np.max(np.abs(V_td0 - V_fwd)):.4f}  (λ={lam}이므로 차이 있음)")


if __name__ == "__main__":
    main()
