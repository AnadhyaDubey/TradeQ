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