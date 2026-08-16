"""
quick_eval.py
================
Fast reality check: run the saved best DQN and DDPG checkpoints on the
UNSEEN test set (2022-2023 data the agents never trained on), with
exploration turned off. This is the real test -- training-set net worth
can be inflated by memorizing a fixed, deterministic training sequence,
but the test set is genuinely new data.
"""

import torch
import numpy as np

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.dqn_agent import QNetwork
from agents.ddpg_agent import ActorNetwork

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


def continuous_to_discrete(action_value: float) -> int:
    if action_value > BUY_THRESHOLD:
        return 1
    elif action_value < SELL_THRESHOLD:
        return 2
    return 0


data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

env = TradingEnv(test_norm, FEATURE_COLUMNS)

start_price = test_norm.iloc[0]["Close"]
end_price = test_norm.iloc[-1]["Close"]
buy_hold_final = 10_000 * (end_price / start_price) * (1 - 0.001)
print(f"Buy-and-hold baseline: ${buy_hold_final:,.2f} "
      f"(AAPL ${start_price:.2f} -> ${end_price:.2f})")

dqn_net = QNetwork(state_dim=env.state_dim, action_dim=env.action_space_n)
dqn_net.load_state_dict(torch.load("models/dqn_best.pth"))
dqn_net.eval()

state = env.reset()
done = False
while not done:
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = int(torch.argmax(dqn_net(state_t)).item())
    state, reward, done, info = env.step(action)
print(f"DQN  (test set):        ${info['net_worth']:,.2f}")

ddpg_actor = ActorNetwork(state_dim=env.state_dim, action_dim=1)
ddpg_actor.load_state_dict(torch.load("models/ddpg_actor_best.pth"))
ddpg_actor.eval()

state = env.reset()
done = False
while not done:
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        raw_action = ddpg_actor(state_t).numpy()[0]
    action = continuous_to_discrete(raw_action[0])
    state, reward, done, info = env.step(action)
print(f"DDPG (test set):        ${info['net_worth']:,.2f}")