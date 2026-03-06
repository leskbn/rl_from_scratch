# python -m scripts.ddpg.run
import os
import json
import gymnasium as gym
import torch
from rl.algos.ddpg import DDPG
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "Hopper-v5"
    save_dir = f"results/ddpg/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    env = gym.make(env_name)
    action_high = env.action_space.high

    # 저장 경로 설정
    rewards_history = []

    # 하이퍼파라미터
    lr_actor = 1e-4
    lr_critic = 1e-3
    gamma = 0.99
    buffer_size = 1_000_000
    batch_size = 64
    hidden_dim = [400, 300]
    num_episodes = 3000
    tau = 0.005

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # DDPG 생성
    ddpg = DDPG(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        action_high=action_high,
    )

    total_ep_reward = 0
    best_avg_reward = -float("inf")
    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0
        while not done:
            # action 선택
            action = ddpg.select_action(state)
            action = action * action_high
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)
            # push
            if terminated or truncated:
                done = True
            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            # update
            ddpg.update()

            # state 업데이트
            state = next_state

            total_ep_reward += reward

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)
        rewards_history.append(total_ep_reward)
        # 최근 10 에피소드 평균으로 best 저장
        if len(rewards_history) >= 10:
            avg_reward = sum(rewards_history[-10:]) / 10
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(
                    ddpg.actor.state_dict(),
                    f"{save_dir}/best_actor.pth",
                )
                print(f"best avg reward: {best_avg_reward}, 저장!")

    torch.save(ddpg.actor.state_dict(), f"{save_dir}/actor.pth")
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
