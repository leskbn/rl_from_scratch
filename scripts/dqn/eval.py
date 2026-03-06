# python -m scripts.dqn.eval
import os
import json
import gymnasium as gym
import torch
from rl.algos.dqn import DQN
from datetime import datetime


def main():
    env_name = "CartPole-v1"
    # 환경 생성
    env = gym.make(env_name, render_mode="human")

    # 하이퍼파라미터
    learning_rate = 1e-3
    gamma = 0.99
    buffer_capacity = 10000
    batch_size = 64
    target_update_freq = 100
    hidden_dim = 64
    num_episodes = 3

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # DQN 생성
    dqn = DQN(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )

    # 학습 모델 불러오기
    dqn.q_network.load_state_dict(
        torch.load(
            "results/dqn/CartPole-v1_20260226_193658/q_network.pth", weights_only=True
        )
    )

    total_ep_reward = 0
    total_steps = 0

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0
        while not done:
            # action 선택
            action = dqn.select_greedy_action(state)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)
            # push
            if terminated or truncated:
                done = True

            # state 업데이트
            state = next_state

            total_ep_reward += reward
            total_steps += 1

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)


if __name__ == "__main__":
    main()
