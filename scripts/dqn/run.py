# python -m scripts.dqn.run
import os
import json
import gymnasium as gym
import torch
from rl.algos.dqn import DQN
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "CartPole-v1"
    save_dir = f"results/dqn/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    env = gym.make(env_name)

    # 저장 경로 설정

    rewards_history = []

    # 하이퍼파라미터
    learning_rate = 1e-3
    gamma = 0.99
    buffer_capacity = 10000
    batch_size = 64
    target_update_freq = 100
    hidden_dim = 64
    num_episodes = 500

    eps_start = 1.0
    eps_end = 0.01
    eps_decay_steps = 10000

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

    total_ep_reward = 0
    total_steps = 0
    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0
        while not done:
            eps = max(
                eps_end,
                eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps,
            )
            # action 선택
            action = dqn.select_action(state, eps)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)
            # push
            if terminated or truncated:
                done = True
            dqn.push(state, action, reward, next_state, done)
            # update
            dqn.update()
            # state 업데이트
            state = next_state

            total_ep_reward += reward
            total_steps += 1

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward, "epsilon = ", eps)
        rewards_history.append(total_ep_reward)

    torch.save(dqn.q_network.state_dict(), f"{save_dir}/q_network.pth")
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
