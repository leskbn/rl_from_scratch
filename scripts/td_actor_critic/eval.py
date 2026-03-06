# python -m scripts.td_actor_critic.eval
import os
import json
import gymnasium as gym
import torch
from rl.algos.td_actor_critic import TDActorCritic
from datetime import datetime


def main():
    env_name = "LunarLander-v3"
    # 환경 생성
    env = gym.make(env_name, render_mode="human")
    num_episodes = 5

    # 하이퍼파라미터
    learning_rate = 1e-3
    gamma = 0.990
    hidden_dim = 128

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    td_actor_critic = TDActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
    )

    # 학습 모델 불러오기
    td_actor_critic.policy_network.load_state_dict(
        torch.load(
            "results/td_actor_critic/LunarLander-v3_20260302_201103/policy_network.pth",
            weights_only=True,
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
            action = td_actor_critic.select_greedy_action(state)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)
            # push
            if terminated or truncated:
                done = True

            # state 업데이트
            state = next_state

            total_ep_reward += reward

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)


if __name__ == "__main__":
    main()
