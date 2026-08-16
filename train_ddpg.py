"""
train_ddpg.py
================
Full training loop for the DDPG agent. Same structure as train_dqn.py,
adapted for DDPG's continuous action + noise-based exploration.

NOTE: sigma raised from 0.1 -> 0.2 and thresholds loosened from 0.33 ->
0.15, addressing the exploration collapse we saw in the smoke test
(actions never left the "hold" zone in 20 steps).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.ddpg_agent import DDPGAgent

NUM_EPISODES = 50
BATCH_SIZE = 64
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


def continuous_to_discrete(action_value: float) -> int:
    if action_value > BUY_THRESHOLD:
        return 1
    elif action_value < SELL_THRESHOLD:
        return 2
    return 0


# --- data ---
data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

# --- env + agent ---
env = TradingEnv(train_norm, FEATURE_COLUMNS)
agent = DDPGAgent(state_dim=env.state_dim, action_dim=1)
agent.noise.sigma = 0.2  # more aggressive exploration than the default

episode_net_worths = []
best_net_worth = -np.inf

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    agent.noise.reset()  # fresh noise process each episode
    done = False
    critic_losses, actor_losses = [], []

    while not done:
        raw_action = agent.act(state)
        discrete_action = continuous_to_discrete(raw_action[0])

        next_state, reward, done, info = env.step(discrete_action)
        agent.remember(state, raw_action, reward, next_state if not done else state, done)

        result = agent.learn(batch_size=BATCH_SIZE)
        if result is not None:
            critic_losses.append(result[0])
            actor_losses.append(result[1])

        state = next_state if not done else state

    final_net_worth = info["net_worth"]
    avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
    avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
    episode_net_worths.append(final_net_worth)

    if final_net_worth > best_net_worth:
        best_net_worth = final_net_worth
        torch.save(agent.actor.state_dict(), "models/ddpg_actor_best.pth")
        torch.save(agent.critic.state_dict(), "models/ddpg_critic_best.pth")

    print(f"Episode {episode:3d}/{NUM_EPISODES}  "
          f"net_worth=${final_net_worth:9.2f}  "
          f"critic_loss={avg_critic_loss:.5f}  "
          f"actor_loss={avg_actor_loss:.5f}")

# --- final save + plot ---
torch.save(agent.actor.state_dict(), "models/ddpg_actor_final.pth")
torch.save(agent.critic.state_dict(), "models/ddpg_critic_final.pth")

plt.figure(figsize=(10, 5))
plt.plot(episode_net_worths)
plt.axhline(y=10_000, color="gray", linestyle="--", label="Starting balance")
plt.xlabel("Episode")
plt.ylabel("Final Net Worth ($)")
plt.title("DDPG Training: Net Worth per Episode")
plt.legend()
plt.tight_layout()
plt.savefig("results/ddpg_training_curve.png")
print("\nSaved reward curve to results/ddpg_training_curve.png")
print(f"Best net worth achieved: ${best_net_worth:.2f}")