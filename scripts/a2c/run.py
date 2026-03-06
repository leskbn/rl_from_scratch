# python -m scripts.a2c.run
import os
import json
import gymnasium as gym
import torch
import numpy as np
from rl.algos.a2c import A2C
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "LunarLander-v3"
    save_dir = f"results/a2c/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    num_envs = 8
    env = gym.make_vec(env_name, num_envs=num_envs)
    total_steps = 1000000

    rewards_history = []

    # 하이퍼파라미터
    learning_rate = 3e-4
    gamma = 0.990
    hidden_dim = 128

    # vector env는 shape이 (num_envs, obs_dim)
    obs_dim = env.observation_space.shape[1]
    n_actions = env.action_space[0].n

    a2c = A2C(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
    )

    best_avg_reward = -float("inf")

    # 학습 루프
    # vector env
    # env 4개 동시에 돌아감
    # env1이 끝나면 → 자동 reset → 계속 돌아감
    # env2는 아직 안 끝남 → 계속 돌아감
    # 에피소드 단위가 아닌 스텝 단위로 루프를 짜는 것이 일반적

    episode_rewards = np.zeros(num_envs)  # 4 대신 num_envs!
    states, _ = env.reset()
    for step in range(total_steps):
        # action 선택
        actions, log_probs = a2c.select_action(states)
        # step은 numpy array를 입력으로 받아서 tensor인 actions를 변환
        next_states, rewards, terminated, truncated, _ = env.step(actions.numpy())

        dones = terminated | truncated

        episode_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                rewards_history.append(episode_rewards[i])
                print(
                    f"episode {len(rewards_history)} total reward: {episode_rewards[i]:.1f}"
                )
                episode_rewards[i] = 0

                # 100 에피소드마다 평균으로 best 저장
                if len(rewards_history) >= 100:
                    avg_reward = sum(rewards_history[-100:]) / 100
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(
                            a2c.policy_network.state_dict(),
                            f"{save_dir}/best_policy_network.pth",
                        )
                        print(f"best model 저장! avg reward: {best_avg_reward:.1f}")

        # update
        a2c.update(states, next_states, rewards, log_probs, dones)

        # state 업데이트
        states = next_states

    torch.save(
        a2c.policy_network.state_dict(),
        f"{save_dir}/policy_network.pth",  # 마지막
    )
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
