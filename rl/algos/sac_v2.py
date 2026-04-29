import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from rl.nets.critic_network import CriticNetwork
from rl.buffers.replay_buffer import ReplayBuffer

# region

# endregion


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=[64, 64], activation=nn.ReLU):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, n_actions)
        # state 마다 다른 std
        self.log_std_layer = nn.Linear(in_dim, n_actions)

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # reparameterization
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)

        # log prob with tanh correction
        log_prob = Normal(mean, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a, log_prob


class SAC:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_dim,
        lr_actor,
        lr_critic,
        lr_alpha,
        gamma,
        tau,
        buffer_size,
        batch_size,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau

        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size, continuous=True)

        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dims=self.hidden_dim,
            activation=nn.ReLU,
        )

        self.critic1 = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic2 = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )

        self.critic1_target = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic2_target = CriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optim = torch.optim.Adam(
            self.critic1.parameters(), lr=self.lr_critic
        )
        self.critic2_optim = torch.optim.Adam(
            self.critic2.parameters(), lr=self.lr_critic
        )

        self.lr_alpha = lr_alpha
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.target_entropy = -n_actions

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state)
        return action.squeeze(0).numpy()

    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            features = self.actor.net(state)
            mean = self.actor.mean_layer(features)
            action = torch.tanh(mean)
        return action.squeeze(0).numpy()

    def soft_update(self, network, target):
        for param, target_param in zip(network.parameters(), target.parameters()):
            target_param.data = (
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            next_q = torch.min(q1_next, q2_next) - self.alpha * next_log_probs

            y = rewards + self.gamma * next_q * (1 - dones)

        critic_loss = (
            (self.critic1(states, actions) - y) ** 2
            + (self.critic2(states, actions) - y) ** 2
        ).mean()

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        new_actions_a, log_probs_a = self.actor(states)

        actor_loss = (
            self.alpha * log_probs_a
            - torch.min(
                self.critic1(states, new_actions_a),
                self.critic2(states, new_actions_a),
            )
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(
            self.log_alpha * (log_probs_a + self.target_entropy).detach()
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
