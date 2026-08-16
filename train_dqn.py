"""
train_dqn.py
================
Quick integration smoke test: connect TradingEnv + DQNAgent for a
handful of real steps, using real AAPL data. Full training loop with
logging comes in Session 5 -- this just proves the two pieces work together.
"""

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

env = TradingEnv(train_norm, FEATURE_COLUMNS)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space_n)

state = env.reset()
for i in range(20):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.remember(state, action, reward, next_state if not done else state, done)
    loss = agent.learn(batch_size=8)
    print(f"step={i} action={action} reward={reward:.4f} net_worth={info['net_worth']:.2f} loss={loss}")
    state = next_state
    if done:
        break