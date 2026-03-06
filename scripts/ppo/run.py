# python -m scripts.ppo.run
import os
import json
import gymnasium as gym
import torch
import torch.nn as nn
from rl.algos.ppo import PPO
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "LunarLander-v3"
    save_dir = f"results/ppo/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 환경 생성
    env = gym.make(env_name)

    rewards_history = []

    # 하이퍼파라미터
    lr = 3e-4
    gamma = 0.990
    lam = 0.95

    hidden_dim = 256
    activation_func = nn.ReLU

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
    n_actions = env.action_space.n

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
        continuous=False,
    )

    best_avg_reward = -float("inf")

    # 학습 루프
    for iteration in range(num_iterations):
        state, _ = env.reset()
        states, actions, rewards, dones, log_probs_old, values = [], [], [], [], [], []
        done = False
        ep_reward = 0
        ep_rewards = []

        # Rollout
        for t in range(T):
            # action 선택
            action, log_prob, value = ppo.select_action(state)
            # step
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs_old.append(log_prob)
            values.append(value)

            # state 업데이트
            state = next_state

            ep_reward += reward

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0
                state, _ = env.reset()  # 수동으로 reset

        _, _, last_value = ppo.select_action(state)
        values.append(last_value)
        returns, advantages = ppo.compute_gae(rewards, values, dones)

        ppo.update(states, actions, log_probs_old, returns, advantages)

        if ep_rewards:
            mean_reward = sum(ep_rewards) / len(ep_rewards)
            rewards_history.append(mean_reward)
            print(
                f"Iteration {iteration} mean reward: {mean_reward:.1f} ({len(ep_rewards)} episodes)"
            )

            # 최근 10 iteration 평균으로 best 저장
            if len(rewards_history) >= 10:
                avg_reward = sum(rewards_history[-10:]) / 10
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(
                        ppo.policy_network.state_dict(),
                        f"{save_dir}/best_policy_network.pth",
                    )
                    print(f"best model 저장! avg reward: {best_avg_reward:.1f}")

    torch.save(
        ppo.policy_network.state_dict(),
        f"{save_dir}/policy_network.pth",  # 마지막
    )
    with open(f"{save_dir}/rewards.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == "__main__":
    main()
