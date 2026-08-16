"""
dqn_agent.py
================
A Deep Q-Network agent implemented in PyTorch. Given a state (our 5-number
vector from TradingEnv), the Q-network predicts a Q-value for each of the
3 possible actions (hold, buy, sell) -- an estimate of "how much future
reward do I expect if I take this action from this state?"
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """The 'brain': state in, one Q-value per action out."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # no activation on the output -- Q-values are unbounded


if __name__ == "__main__":
    net = QNetwork(state_dim=5, action_dim=3)
    dummy_state = torch.rand(1, 5)  # 1 fake state, 5 features
    q_values = net(dummy_state)
    print("Q-values:", q_values)
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Stores past experiences; lets us sample a random batch to learn from."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, gamma: float = 0.95):
        self.action_dim = action_dim
        self.gamma = gamma  # how much future reward matters vs immediate reward

        self.epsilon = 1.0       # start fully random
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.update_every = 100   # sync target network every N training steps
        self.train_step_count = 0

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def act(self, state):
        """Epsilon-greedy: explore randomly, or exploit the network's best guess."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def learn(self, batch_size: int = 64):
        """One training step: sample a batch, compute Bellman targets, backprop."""
        if len(self.replay_buffer) < batch_size:
            return None  # not enough experience yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # current Q-values for the actions actually taken
        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: reward + gamma * best next Q-value (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


if __name__ == "__main__":
    agent = DQNAgent(state_dim=5, action_dim=3)

    # Fill the buffer with enough random fake experience to allow a batch
    for i in range(200):
        s = np.random.rand(5).astype(np.float32)
        a = random.randrange(3)
        r = np.random.uniform(-0.02, 0.02)
        s2 = np.random.rand(5).astype(np.float32)
        d = False
        agent.remember(s, a, r, s2, d)

    print("Buffer filled:", len(agent.replay_buffer))

    for step in range(10):
        loss = agent.learn(batch_size=64)
        print(f"Learn step {step}: loss={loss:.5f}" if loss is not None else "not enough data yet")


if __name__ == "__main__":
    agent = DQNAgent(state_dim=5, action_dim=3)
    dummy_state = np.random.rand(5).astype(np.float32)

    for i in range(5):
        action = agent.act(dummy_state)
        print(f"Step {i}: epsilon={agent.epsilon:.3f}  chosen action={action}")
        agent.remember(dummy_state, action, reward=0.01, next_state=dummy_state, done=False)

    print("Replay buffer size:", len(agent.replay_buffer))