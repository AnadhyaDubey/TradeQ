"""
evaluate.py
================
The real backtest: run DQN, DDPG, buy-and-hold, and a random-action
baseline on the held-out test set, track full net-worth trajectories
(not just final values), compute Sharpe ratio / max drawdown / cumulative
return for each, and plot them together for the README.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.data_loader import (
    download_price_data, engineer_features,
    chronological_split, normalize_with_train_stats, FEATURE_COLUMNS,
)
from envs.trading_env import TradingEnv
from agents.dqn_agent import QNetwork
from agents.ddpg_agent import ActorNetwork
from utils.metrics import cumulative_return, sharpe_ratio, max_drawdown

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


def continuous_to_discrete(action_value: float) -> int:
    if action_value > BUY_THRESHOLD:
        return 1
    elif action_value < SELL_THRESHOLD:
        return 2
    return 0


def run_dqn(env, model_path):
    net = QNetwork(state_dim=env.state_dim, action_dim=env.action_space_n)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(net(state_t)).item())
        state, reward, done, info = env.step(action)
    return env.net_worth_history


def run_ddpg(env, model_path):
    actor = ActorNetwork(state_dim=env.state_dim, action_dim=1)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            raw_action = actor(state_t).numpy()[0]
        action = continuous_to_discrete(raw_action[0])
        state, reward, done, info = env.step(action)
    return env.net_worth_history


def run_buy_and_hold(env):
    state = env.reset()
    env.step(1)  # buy once on day 1
    done = False
    while not done:
        state, reward, done, info = env.step(0)  # hold every day after
    return env.net_worth_history


def run_random_baseline(env, n_seeds: int = 20):
    """
    Average results over many random seeds instead of trusting one lucky
    (or unlucky) draw. This is the statistically honest way to report a
    random baseline -- a single rollout tells you almost nothing.
    """
    all_histories = []
    for seed in range(n_seeds):
        random.seed(seed)
        state = env.reset()
        done = False
        while not done:
            action = random.randrange(3)
            state, reward, done, info = env.step(action)
        all_histories.append(env.net_worth_history)

    # average net worth at each time step, across all seeds
    avg_history = np.mean(np.array(all_histories), axis=0).tolist()
    return avg_history

def report(name, history):
    cr = cumulative_return(history)
    sr = sharpe_ratio(history)
    mdd = max_drawdown(history)
    print(f"{name:20s}  final=${history[-1]:>10,.2f}  "
          f"return={cr*100:>7.2f}%  sharpe={sr:>6.2f}  max_drawdown={mdd*100:>7.2f}%")
    return {"name": name, "history": history, "cumulative_return": cr,
            "sharpe": sr, "max_drawdown": mdd}


# --- data ---
data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
featured = engineer_features(data)
train, test = chronological_split(featured, "2022-01-01")
cols = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols)

env = TradingEnv(test_norm, FEATURE_COLUMNS)

print(f"Evaluating on test set: {test_norm.index.min().date()} to {test_norm.index.max().date()}\n")

results = []
results.append(report("Buy-and-hold", run_buy_and_hold(env)))
results.append(report("Random actions (avg of 20 seeds)", run_random_baseline(env)))
results.append(report("DQN", run_dqn(env, "models/dqn_best.pth")))
results.append(report("DDPG", run_ddpg(env, "models/ddpg_actor_best.pth")))

# --- plot all trajectories together ---
plt.figure(figsize=(12, 6))
for r in results:
    plt.plot(r["history"], label=r["name"])
plt.axhline(y=10_000, color="gray", linestyle="--", alpha=0.5, label="Starting balance")
plt.xlabel("Trading day (test period)")
plt.ylabel("Net Worth ($)")
plt.title("Backtest: Net Worth on Held-Out Test Set (2022-2023)")
plt.legend()
plt.tight_layout()
plt.savefig("results/backtest_comparison.png")
print("\nSaved comparison plot to results/backtest_comparison.png")