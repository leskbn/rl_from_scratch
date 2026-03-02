import torch
import torch.nn as nn
import numpy as np
from rl.nets.policy_network import PolicyNetwork
from rl.nets.value_network import ValueNetwork


class ActorCritic:
    def __init__(self, obs_dim, n_actions, hidden_dim, lr, gamma):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma

        # Policy Network (Actor)
        self.policy_network = PolicyNetwork(
            self.obs_dim, self.n_actions, self.hidden_dim
        )

        # Value Network (Critic)
        self.value_network = ValueNetwork(self.obs_dim, self.hidden_dim)

        # Optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.lr
        )

    def select_action(self, state):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def select_greedy_action(self, state):
        # 확률분포에서 샘플링
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        action = probs.argmax().item()

        return action

    def update(self, states, rewards, log_probs):
        # Advantage, return 계산
        A = 0
        G = 0
        returns = []
        advantages = []

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        for t in range(len(states)):
            # value_network 출력이 (1, 1) shape이라서 squeeze
            state_tensor = torch.tensor(states[t], dtype=torch.float32).unsqueeze(0)
            V = self.value_network(state_tensor).squeeze()
            A = returns[t] - V
            advantages.append(A)

        # loss 계산 및 역전파
        actor_loss = 0
        critic_loss = 0
        for t in range(len(log_probs)):
            actor_loss += -advantages[t] * log_probs[t]
            critic_loss += advantages[t] ** 2

        self.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
