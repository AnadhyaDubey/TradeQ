"""
train_ddpg.py
================
Integration smoke test: connect TradingEnv + DDPGAgent for a handful of
real steps. DDPG outputs a continuous value in [-1, 1]; we convert it to
a discrete hold/buy/sell action here so TradingEnv doesn't need to know
or care which kind of agent is driving it.
"""

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.ddpg_agent import DDPGAgent


def continuous_to_discrete(action_value: float, buy_threshold: float = 0.33, sell_threshold: float = -0.33) -> int:
    """Map a continuous [-1, 1] action into hold/buy/sell."""
    if action_value > buy_threshold:
        return 1  # buy
    elif action_value < sell_threshold:
        return 2  # sell
    return 0  # hold


data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

env = TradingEnv(train_norm, FEATURE_COLUMNS)
agent = DDPGAgent(state_dim=env.state_dim, action_dim=1)

state = env.reset()
for i in range(20):
    raw_action = agent.act(state)
    discrete_action = continuous_to_discrete(raw_action[0])

    next_state, reward, done, info = env.step(discrete_action)
    agent.remember(state, raw_action, reward, next_state if not done else state, done)
    result = agent.learn(batch_size=8)

    loss_str = f"critic={result[0]:.5f} actor={result[1]:.5f}" if result else "no update yet"
    print(f"step={i} raw={raw_action[0]:.3f} discrete={discrete_action} reward={reward:.4f} net_worth={info['net_worth']:.2f} {loss_str}")

    state = next_state
    if done:
        break