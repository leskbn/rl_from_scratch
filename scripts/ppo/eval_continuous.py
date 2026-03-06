# python -m scripts.ppo.eval_continuous
import os
import json
import gymnasium as gym
import torch
import torch.nn as nn
from rl.algos.ppo import PPO
from datetime import datetime


def main():
    env_name = "Walker2d-v5"
    # 환경 생성
    env = gym.make(env_name, render_mode="human")

    num_episodes = 5

    rewards_history = []

    # 하이퍼파라미터
    lr = 3e-4
    gamma = 0.990
    lam = 0.95

    hidden_dim = [64, 64]
    activation_func = nn.Tanh

    eps_clip = 0.2  # clipping 범위
    K_epochs = 10  # 반복 업데이트 횟수
    minibatch_size = 64

    # loss 가중치
    c1 = 0.5  # critic loss 가중치
    c2 = 0.01  # entropy bonus 가중치

    # rollout
    T = 2048  # 한 번에 모을 스텝 수
    num_iterations = 500  # 총 iterations (총 스텝 = T * num_iterations = 1,024,000)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
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

    # 학습 모델 불러오기
    ppo.policy_network.load_state_dict(
        torch.load(
            "results/ppo/Walker2d-v5_20260305_155113/best_policy_network.pth",
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
            action = ppo.select_greedy_action(state)
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
