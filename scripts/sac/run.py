# python -m scripts.sac.run
import os
import json
import gymnasium as gym
import torch
from rl.algos.sac import SAC
from datetime import datetime


def evaluate(sac, env, action_high, n_episodes=5):
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = sac.select_greedy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action * action_high
            )
            done = terminated or truncated
            state = next_state
            ep_reward += reward
        total_rewards.append(ep_reward)
    return sum(total_rewards) / len(total_rewards)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = "Ant-v5"
    save_dir = f"results/sac/{env_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    action_high = env.action_space.high

    train_rewards_history = []
    eval_rewards_history = []

    # 하이퍼파라미터
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_value = 3e-4
    gamma = 0.99
    buffer_size = 1_000_000
    batch_size = 256
    hidden_dim = [256, 256]
    max_steps = 3_000_000
    warmup_steps = 25_000
    tau = 0.005
    alpha = 1.0
    reward_scale = 5

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    sac = SAC(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_value=lr_value,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        alpha=alpha,
        reward_scale=reward_scale,
    )

    # Warm-up: 랜덤 action으로 replay buffer 채우기
    print(f"warming up {warmup_steps} steps...")
    state, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample() / action_high  # -1~1로 정규화
        next_state, reward, terminated, truncated, _ = env.step(action * action_high)
        done = terminated or truncated
        sac.replay_buffer.push(state, action, reward, next_state, done)
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
            action = sac.select_action(state)
            action_scaled = action * action_high

            next_state, reward, terminated, truncated, _ = env.step(action_scaled)
            done = terminated or truncated

            sac.replay_buffer.push(state, action, reward, next_state, done)

            sac.update()

            state = next_state

            total_ep_reward += reward
            currnet_step += 1

            if currnet_step % 5000 == 0:
                eval_reward = evaluate(sac, eval_env, action_high)
                eval_rewards_history.append((currnet_step, eval_reward))
                print(f"[EVAL] steps {currnet_step} avg reward: {eval_reward:.1f}")
                if eval_reward > best_avg_reward:
                    best_avg_reward = eval_reward
                    torch.save(sac.actor.state_dict(), f"{save_dir}/best_actor.pth")
                    print(f"best eval reward: {best_avg_reward:.1f}, 저장!")

        print(f"steps {currnet_step} total reward: {total_ep_reward}")
        train_rewards_history.append(total_ep_reward)

    torch.save(sac.actor.state_dict(), f"{save_dir}/actor.pth")
    with open(f"{save_dir}/train_rewards.json", "w") as f:
        json.dump(train_rewards_history, f)
    with open(f"{save_dir}/eval_rewards.json", "w") as f:
        json.dump(eval_rewards_history, f)


if __name__ == "__main__":
    main()
