"""
trading_env.py
================
A custom trading environment following the standard Gym/Gymnasium
interface: reset() starts a new episode and returns the first
observation; step(action) takes one action, advances time by one day,
and returns (next_state, reward, done, info).
"""

import numpy as np
from data.data_loader import (
    download_price_data,
    engineer_features,
    chronological_split,
    normalize_with_train_stats,
    FEATURE_COLUMNS,
)


class TradingEnv:
    def __init__(self, df, feature_cols, initial_balance: float = 10_000.0, transaction_cost: float = 0.001):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.action_space_n = 3  # 0 = hold, 1 = buy, 2 = sell
        self.state_dim = len(feature_cols) + 1  # +1 for "am I currently holding shares"

    def reset(self):
        """Start a new episode at the beginning of the dataframe."""
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares_held = 0.0
        self.position = 0  # 0 = flat, 1 = holding
        self.net_worth_history = [self.initial_balance]
        return self._get_state()

    def _get_state(self):
        """Build the observation vector: today's features + current position."""
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)
        position_flag = np.array([self.position], dtype=np.float32)
        return np.concatenate([features, position_flag])

    def step(self, action: int):
        """
        action: 0 = hold, 1 = buy, 2 = sell
        Returns (next_state, reward, done, info)
        """
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.cash + self.shares_held * current_price

        if action == 1 and self.position == 0:
            cost = self.cash * self.transaction_cost
            spendable = self.cash - cost
            self.shares_held = spendable / current_price
            self.cash = 0.0
            self.position = 1

        elif action == 2 and self.position == 1:
            proceeds = self.shares_held * current_price
            cost = proceeds * self.transaction_cost
            self.cash = proceeds - cost
            self.shares_held = 0.0
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = self.df.iloc[self.current_step]["Close"]
        new_net_worth = self.cash + self.shares_held * next_price
        reward = (new_net_worth - prev_net_worth) / prev_net_worth

        self.net_worth_history.append(new_net_worth)

        next_state = self._get_state() if not done else None
        info = {"net_worth": new_net_worth, "cash": self.cash, "shares_held": self.shares_held}

        return next_state, reward, done, info


if __name__ == "__main__":
    data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
    featured = engineer_features(data)
    train, test = chronological_split(featured, "2022-01-01")
    cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
    train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

    env = TradingEnv(train_norm, FEATURE_COLUMNS)
    state = env.reset()
    print("Initial state:", state)

    actions_to_try = [1, 0, 0, 2, 1, 0]  # buy, hold, hold, sell, buy, hold
    for a in actions_to_try:
        state, reward, done, info = env.step(a)
        print(f"action={a}  reward={reward:.5f}  net_worth={info['net_worth']:.2f}  done={done}")