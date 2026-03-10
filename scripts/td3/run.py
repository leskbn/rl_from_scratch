# python -m scripts.td3.run
import os
import json
import gymnasium as gym
import torch
from rl.algos.td3 import TD3
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "Walker2d-v5"
    save_dir = f"results/td3/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)
    action_high = env.action_space.high

    rewards_history = []

    # 하이퍼파라미터
    lr_actor = 3e-4
    lr_critic = 3e-4
    gamma = 0.99
    buffer_size = 1_000_000
    batch_size = 256
    hidden_dim = [256, 256]
    max_steps = 1_000_000
    warmup_steps = 25_000
    tau = 0.005
    policy_freq = 2
    noise_clip = 0.5
    noise_std = 0.2

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    td3 = TD3(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_freq=policy_freq,
        noise_clip=noise_clip,
        noise_std=noise_std,
        action_high=action_high,
    )

    # Warm-up: 랜덤 action으로 replay buffer 채우기
    print(f"warming up {warmup_steps} steps...")
    state, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample() / action_high  # -1~1로 정규화
        next_state, reward, terminated, truncated, _ = env.step(action * action_high)
        done = terminated or truncated
        td3.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
    print("warming up done!")

    # 학습 루프
    currnet_step = 0
    best_avg_reward = -float("inf")

    while currnet_step < max_steps:
        state, _ = env.reset()
        done = False
        total_ep_reward = 0

        while not done:
            action = td3.select_action(state)
            action_scaled = action * action_high

            next_state, reward, terminated, truncated, _ = env.step(action_scaled)
            done = terminated or truncated

            td3.replay_buffer.push(state, action, reward, next_state, done)

            td3.update()

            state = next_state

            total_ep_reward += reward
            currnet_step += 1

        print(f"steps {currnet_step} total reward: {total_ep_reward}")
        rewards_history.append(total_ep_reward)

        # 최근 10 에피소드 평균으로 best 저장
        if len(rewards_history) >= 10:
            avg_reward = sum(rewards_history[-10:]) / 10
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(td3.actor.state_dict(), f"{save_dir}/best_actor.pth")
                print(f"best avg reward: {best_avg_reward:.1f}, 저장!")

    torch.save(td3.actor.state_dict(), f"{save_dir}/actor.pth")
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
