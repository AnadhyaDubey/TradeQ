# TradeQ: Reinforcement Learning for Algorithmic Trading

## Overview

This project trains two reinforcement learning agents — **DQN** (discrete actions) and **DDPG** (continuous actions) — to trade AAPL stock, then rigorously backtests them against buy-and-hold and randomized baselines on a held-out test period they never saw during training. It's built end-to-end from scratch: a real market data pipeline, a custom Gym-style trading environment, two agents implemented in PyTorch, and a full evaluation suite using industry-standard risk metrics (Sharpe ratio, max drawdown).

The goal wasn't just "make an agent that trades" — it was to build something methodologically honest: catching and fixing real issues like lookahead bias, training-data memorization, and a misleading baseline along the way (see **Challenges & Lessons Learned** below).

## What's In This Project

- **A real financial data pipeline** — downloads split-adjusted OHLCV data via `yfinance`, engineers 4 technical-indicator features (log returns, RSI, moving-average ratio, rolling volatility), and splits/normalizes it in a way that avoids lookahead bias
- **A custom RL trading environment** — built to the standard Gym interface (`reset()` / `step()`), with portfolio bookkeeping (cash, shares held, transaction costs) and reward shaped as daily portfolio return
- **Two RL agents implemented in PyTorch from first principles** — DQN (Q-network, replay buffer, epsilon-greedy exploration, target network, Bellman-equation updates) and DDPG (actor-critic architecture, Ornstein-Uhlenbeck exploration noise, soft target updates) — including bug fixes and vectorization improvements over the original reference implementation this project started from
- **A rigorous backtest suite** — Sharpe ratio, max drawdown, and cumulative return, benchmarked against buy-and-hold and a properly averaged (20-seed) random-action baseline, not just a single lucky run

## Key Results

Evaluated on a held-out test set (2022–2023) the agents never saw during training:

| Strategy | Final Value | Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| Buy-and-hold | $10,688 | +6.88% | 0.26 | -30.98% |
| Random actions (avg of 20 seeds) | $9,044 | -9.56% | -0.27 | -24.30% |
| DQN | $8,686 | -13.14% | -0.23 | -31.88% |
| **DDPG** | **$12,255** | **+22.55%** | **0.57** | **-20.20%** |

**DDPG beat buy-and-hold on both return and risk-adjusted performance**, while **DQN underperformed even a properly-averaged random baseline** — a genuine, reported finding about discrete vs. continuous action formulations for this task, not a cherry-picked result.

*(Backtest comparison plot available in `results/backtest_comparison.png`)*

## How It Works
TradeQ/
├── data/
│ └── data_loader.py # download, feature engineering, chronological split
├── envs/
│ └── trading_env.py # Gym-style TradingEnv (reset/step)
├── agents/
│ ├── dqn_agent.py # QNetwork, ReplayBuffer, epsilon-greedy, Bellman update
│ └── ddpg_agent.py # Actor/Critic networks, OU noise, soft target updates
├── utils/
│ └── metrics.py # Sharpe ratio, max drawdown, cumulative return
├── models/ # saved checkpoints (gitignored -- large binary files)
├── results/ # training curves, backtest plots
├── train_dqn.py # full DQN training loop
├── train_ddpg.py # full DDPG training loop
├── evaluate.py # backtest all strategies on held-out test data
└── quick_eval.py # fast single-run sanity check


## Setup

```bash
git clone https://github.com/AnadhyaDubey/TradeQ
cd TradeQ
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

## Running

```bash
python train_dqn.py      # trains DQN, saves models/dqn_best.pth, results/dqn_training_curve.png
python train_ddpg.py     # trains DDPG, saves models/ddpg_actor_best.pth, results/ddpg_training_curve.png
python evaluate.py       # backtests all 4 strategies on the test set, saves results/backtest_comparison.png
```

## Challenges & Lessons Learned

**Lookahead bias in the data pipeline.** Time-series data can't be split or normalized like typical ML datasets. Splitting chronologically (train strictly before test, no shuffling) and computing normalization statistics *only* from the training set — then applying those same stats to test — was essential to avoid the model implicitly "seeing the future."

**Training-curve inflation from a deterministic environment.** During initial training, DDPG's net worth appeared to grow 73x over 50 episodes. This looked like a huge success but was actually a red flag: since every episode replays the exact same historical price sequence in the same order, the agent can partially *memorize* that specific sequence rather than learning a generalizable strategy. Evaluating on the untouched 2022–2023 test set gave the real, far more modest (and far more credible) number: +22.55%. This distinction — training performance vs. genuine held-out generalization — became the central methodological lesson of the project.

**A single-seed random baseline was misleadingly weak.** An early evaluation had random actions *beating* buy-and-hold, purely because one lucky random seed happened to sidestep the 2022 bear-market drawdown. Averaging net worth trajectories across 20 random seeds gave a statistically honest baseline (-9.56% average return), which correctly showed random performing worse than buy-and-hold — resolving what initially looked like a nonsensical result.

**Action-space mismatch bug in the original DDPG implementation.** The starting codebase clipped actions to `[0, 2]` while the actor network's `tanh` output naturally produces `[-1, 1]` — a silent bug that would have corrupted every trading decision. Fixed by aligning the clip range to match the network's actual output range.

**Non-vectorized replay updates.** The original DDPG `experience_replay` looped over each sample in a batch individually rather than processing the batch as a single tensor operation — both slower and not representative of standard PyTorch practice. Rewrote it as a fully vectorized batch update.

**DQN underperformed DDPG substantially.** Rather than discard this result, it's reported directly: DQN's discrete hold/buy/sell menu appears less able to generalize on this task than DDPG's continuous position-sizing, converted to discrete action via thresholding. A natural next experiment (see Future Work) is letting DDPG's continuous output control position *size* directly, rather than being bucketed into 3 discrete actions.

## Future Improvements

- Let DDPG's continuous action directly control position size (0-100% of capital) instead of thresholding into 3 discrete buckets
- Train across multiple tickers instead of a single stock, to test whether the strategy generalizes across assets
- Add more engineered features (volume, MACD, Bollinger Bands)
- Hyperparameter tuning (learning rates, network depth, reward shaping)
- Docker containerization and a FastAPI inference service (in progress)

## License

MIT
