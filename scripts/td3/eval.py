# python -m scripts.td3.eval
import os
import json
import gymnasium as gym
import torch
from rl.algos.td3 import TD3
from datetime import datetime


def main():
    env_name = "Walker2d-v5"

    # 환경 생성
    env = gym.make(env_name, render_mode="human")
    action_high = env.action_space.high

    # 저장 경로 설정
    rewards_history = []

    # 하이퍼파라미터
    lr_actor = 3e-4
    lr_critic = 3e-4
    gamma = 0.99
    buffer_size = 1_000_000
    batch_size = 256
    hidden_dim = [256, 256]
    num_episodes = 5
    tau = 0.005
    policy_freq = 2
    noise_clip = 0.5
    noise_std = 0.2

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # TD3 생성
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
    td3.actor.load_state_dict(
        torch.load(
            "results/td3/Walker2d-v5_20260306_211745/best_actor.pth", weights_only=True
        )
    )

    total_ep_reward = 0

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0
        while not done:
            # action 선택
            action = td3.select_greedy_action(state)
            action = action * action_high
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                done = True

            # state 업데이트
            state = next_state

            total_ep_reward += reward

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)
        rewards_history.append(total_ep_reward)


if __name__ == "__main__":
    main()
