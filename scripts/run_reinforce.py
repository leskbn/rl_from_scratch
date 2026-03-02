# python -m scripts.run_reinforce
import os
import json
import gymnasium as gym
import torch
from rl.algos.reinforce import REINFORCE
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "CartPole-v1"
    save_dir = f"results/reinforce/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    env = gym.make(env_name)
    num_episodes = 1000

    # 저장 경로 설정

    rewards_history = []

    # 하이퍼파라미터
    learning_rate = 1e-3
    gamma = 0.990
    hidden_dim = 64

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    reinforce = REINFORCE(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
    )

    total_ep_reward = 0
    best_reward = 0

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0

        rewards = []
        log_probs = []
        while not done:
            # action 선택
            action, log_prob = reinforce.select_action(state)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            if terminated or truncated:
                done = True

            rewards.append(reward)
            log_probs.append(log_prob)

            # state 업데이트
            state = next_state

            total_ep_reward += reward

        # update
        reinforce.update(rewards, log_probs)

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)
        rewards_history.append(total_ep_reward)

        # variance가 높기 때문에 학습 중간에 최고 점수를 받은 모델을 저장
        if total_ep_reward > best_reward:
            best_reward = total_ep_reward
            torch.save(
                reinforce.policy_network.state_dict(),
                f"{save_dir}/policy_network.pth",
            )

    torch.save(
        reinforce.policy_network.state_dict(),
        f"{save_dir}/policy_network.pth",  # 마지막
    )
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
