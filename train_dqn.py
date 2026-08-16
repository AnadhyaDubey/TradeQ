"""
train_dqn.py
================
Full training loop for the DQN agent: many episodes over the training
data, epsilon decay, periodic logging, reward-curve plotting, and saving
the trained model weights to disk.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

NUM_EPISODES = 50
BATCH_SIZE = 64

# --- data ---
data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

# --- env + agent ---
env = TradingEnv(train_norm, FEATURE_COLUMNS)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space_n)

episode_net_worths = []
episode_losses = []
best_net_worth = -np.inf

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    done = False
    losses_this_episode = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state if not done else state, done)

        loss = agent.learn(batch_size=BATCH_SIZE)
        if loss is not None:
            losses_this_episode.append(loss)

        state = next_state if not done else state

    agent.decay_epsilon()

    final_net_worth = info["net_worth"]
    avg_loss = np.mean(losses_this_episode) if losses_this_episode else 0.0
    episode_net_worths.append(final_net_worth)
    episode_losses.append(avg_loss)

    if final_net_worth > best_net_worth:
        best_net_worth = final_net_worth
        torch.save(agent.q_network.state_dict(), "models/dqn_best.pth")

    print(f"Episode {episode:3d}/{NUM_EPISODES}  "
          f"net_worth=${final_net_worth:9.2f}  "
          f"epsilon={agent.epsilon:.3f}  "
          f"avg_loss={avg_loss:.5f}")

# --- final save + plot ---
torch.save(agent.q_network.state_dict(), "models/dqn_final.pth")

plt.figure(figsize=(10, 5))
plt.plot(episode_net_worths)
plt.axhline(y=10_000, color="gray", linestyle="--", label="Starting balance")
plt.xlabel("Episode")
plt.ylabel("Final Net Worth ($)")
plt.title("DQN Training: Net Worth per Episode")
plt.legend()
plt.tight_layout()
plt.savefig("results/dqn_training_curve.png")
print("\nSaved reward curve to results/dqn_training_curve.png")
print(f"Best net worth achieved: ${best_net_worth:.2f}")