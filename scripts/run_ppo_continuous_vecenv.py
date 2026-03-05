# python -m scripts.run_ppo_continuous_vecenv
import os
import json
import gymnasium as gym
import torch
import torch.nn as nn
from rl.algos.ppo import PPO
import numpy as np
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "Walker2d-v5"
    save_dir = f"results/ppo/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    num_envs = 8
    env = gym.make_vec(env_name, num_envs=num_envs)
    rewards_history = []

    # 하이퍼파라미터
    lr = 3e-4
    gamma = 0.99
    lam = 0.95

    hidden_dim = [64, 64]
    activation_func = nn.Tanh

    eps_clip = 0.2
    K_epochs = 10
    minibatch_size = 64

    c1 = 0.5
    c2 = 0.01

    T = 2048
    num_iterations = 500

    obs_dim = env.observation_space.shape[1]
    n_actions = env.action_space.shape[1]
    action_low = env.action_space.low
    action_high = env.action_space.high

    ppo = PPO(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        activation_func=activation_func,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        K_epochs=K_epochs,
        lam=lam,
        c1=c1,
        c2=c2,
        minibatch_size=minibatch_size,
        continuous=True,
        action_high=action_high,
        action_low=action_low,
    )

    best_avg_reward = -float("inf")
    total_steps = 0

    for iteration in range(num_iterations):
        states, _ = env.reset()
        states_buf, actions_buf, rewards_buf, dones_buf, log_probs_buf, values_buf = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        ep_rewards = []
        episode_rewards = np.zeros(num_envs)  # 각 환경의 누적 reward

        # Rollout
        for t in range(T):
            # action
            action, log_prob, value = ppo.select_action(states)
            action_clipped = np.clip(action, action_low, action_high)

            # step
            next_states, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated | truncated

            # 버퍼에 저장
            states_buf.append(states)
            actions_buf.append(action)
            rewards_buf.append(reward)
            dones_buf.append(done)
            log_probs_buf.append(log_prob)
            values_buf.append(value)

            # 에피소드 reward 추적
            episode_rewards += reward
            for i, d in enumerate(done):
                if d:
                    ep_rewards.append(float(episode_rewards[i]))
                    episode_rewards[i] = 0

            states = next_states
            total_steps += num_envs

        _, _, last_value = ppo.select_action(states)
        values_buf.append(last_value)

        # GAE 계산 — 버퍼를 (T * num_envs,) 형태로 flatten
        states_arr = np.array(states_buf).reshape(-1, obs_dim)  # (T*num_envs, obs_dim)
        actions_arr = np.array(actions_buf).reshape(
            -1, n_actions
        )  # (T*num_envs, n_actions)
        rewards_arr = np.array(rewards_buf).flatten()  # (T*num_envs,)
        dones_arr = np.array(dones_buf).flatten()  # (T*num_envs,)
        log_probs_arr = np.array(log_probs_buf).flatten()  # (T*num_envs,)
        values_arr = np.array(values_buf).flatten()  # (T*num_envs + num_envs,)

        all_returns = np.zeros(T * num_envs)
        all_advantages = np.zeros(T * num_envs)

        for i in range(num_envs):
            env_rewards = np.array(rewards_buf)[:, i]
            env_values = np.array(values_buf)[:, i]
            env_dones = np.array(dones_buf)[:, i]

            ret, adv = ppo.compute_gae(
                env_rewards.tolist(), env_values.tolist(), env_dones.tolist()
            )
            all_returns[i::num_envs] = ret  # env i 위치에 저장
            all_advantages[i::num_envs] = adv

        ppo.update(
            states_arr,
            actions_arr,
            log_probs_arr,
            all_returns.tolist(),
            all_advantages.tolist(),
        )

        if ep_rewards:
            mean_reward = sum(ep_rewards) / len(ep_rewards)
            rewards_history.append(mean_reward)
            print(
                f"Iteration {iteration} mean reward: {mean_reward:.1f} ({len(ep_rewards)} episodes) steps: {total_steps}"
            )

            if len(rewards_history) >= 10:
                avg_reward = sum(rewards_history[-10:]) / 10
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(
                        ppo.policy_network.state_dict(),
                        f"{save_dir}/best_policy_network.pth",
                    )
                    print(f"best model 저장! avg reward: {best_avg_reward:.1f}")

    torch.save(
        ppo.policy_network.state_dict(),
        f"{save_dir}/policy_network.pth",
    )
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
