# python -m scripts.td_actor_critic.run
import os
import json
import gymnasium as gym
import torch
from rl.algos.td_actor_critic import TDActorCritic
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "LunarLander-v3"
    save_dir = f"results/td_actor_critic/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    env = gym.make(env_name)
    num_episodes = 3000

    rewards_history = []

    # 하이퍼파라미터
    learning_rate = 3e-4
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

    total_ep_reward = 0
    best_avg_reward = -float("inf")

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_ep_reward = 0

        while not done:
            # action 선택
            action, log_prob = td_actor_critic.select_action(state)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            if terminated or truncated:
                done = True

            # update
            td_actor_critic.update(state, next_state, reward, log_prob, done)

            # state 업데이트
            state = next_state

            total_ep_reward += reward

        # 에피소드 결과 출력
        print(f"episode {episode} total reward: ", total_ep_reward)
        rewards_history.append(total_ep_reward)

        # 학습 중 최고 점수를 받은 모델을 저장
        if len(rewards_history) >= 100:
            avg_reward = sum(rewards_history[-100:]) / 100

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(
                    td_actor_critic.policy_network.state_dict(),
                    f"{save_dir}/best_policy_network.pth",
                )
                print(f"best model 저장! avg reward: {best_avg_reward:.1f}")

    torch.save(
        td_actor_critic.policy_network.state_dict(),
        f"{save_dir}/policy_network.pth",  # 마지막
    )
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
