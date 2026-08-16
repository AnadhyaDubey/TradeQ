"""
ddpg_agent.py
================
DDPG (Deep Deterministic Policy Gradient) -- ported and bug-fixed from
the original ddpg_agent_py.py. Unlike DQN (which scores a fixed menu of
3 discrete actions), DDPG outputs ONE continuous number representing
desired position size, from -1 (fully short/sold) to +1 (fully bought
in). This is the "dial" instead of "menu" distinction discussed earlier.

BUGS FIXED FROM THE ORIGINAL:
1. act() used to clip actions to [0, 2], but the actor's tanh output is
   naturally [-1, 1] -- clipping to the wrong range silently distorted
   every action. Now clip matches tanh's actual range.
2. experience_replay looped over the batch one sample at a time (slow,
   and not truly "batch" training). Now fully vectorized -- the whole
   batch is processed in a single forward/backward pass.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Given a state, directly outputs the chosen action (the 'dial position')."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # bounded to [-1, 1]


class CriticNetwork(nn.Module):
    """Given a (state, action) pair, estimates how good that action was."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # unbounded Q-value estimate


class OrnsteinUhlenbeckNoise:
    """Correlated random noise for exploration -- deterministic actors need this."""

    def __init__(self, size, mu=0.0, sigma=0.1, theta=0.15):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.size = size
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int = 1, gamma: float = 0.95, tau: float = 0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # soft update rate -- much faster than the original's 0.0001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise = OrnsteinUhlenbeckNoise(size=action_dim)

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer()

    def act(self, state, add_noise: bool = True):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if add_noise:
            action = action + self.noise.sample()
        # FIXED: clip to [-1, 1] to match the actor's actual tanh output range
        return np.clip(action, -1.0, 1.0)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).unsqueeze(1).to(self.device)

        # --- critic update (whole batch at once -- FIXED, was a per-sample loop) ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + self.gamma * target_q * (1 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- actor update: maximize the critic's judgment of our chosen actions ---
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- soft update both target networks ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


if __name__ == "__main__":
    agent = DDPGAgent(state_dim=5, action_dim=1)
    dummy_state = np.random.rand(5).astype(np.float32)

    action = agent.act(dummy_state)
    print("Sampled action (should be in [-1, 1]):", action)

    for i in range(200):
        s = np.random.rand(5).astype(np.float32)
        a = np.random.uniform(-1, 1, size=1).astype(np.float32)
        r = np.random.uniform(-0.02, 0.02)
        s2 = np.random.rand(5).astype(np.float32)
        agent.remember(s, a, r, s2, False)

    for step in range(5):
        result = agent.learn(batch_size=64)
        if result:
            critic_loss, actor_loss = result
            print(f"step={step} critic_loss={critic_loss:.5f} actor_loss={actor_loss:.5f}")