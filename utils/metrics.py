"""
metrics.py
================
Standard finance backtest metrics: cumulative return, Sharpe ratio, max
drawdown. These are what turn "my agent made money" into a credible,
risk-adjusted, comparable result -- the numbers that actually belong on
a resume.
"""

import numpy as np


def cumulative_return(net_worth_history: list) -> float:
    """Total % gain/loss from start to end of the trajectory."""
    start = net_worth_history[0]
    end = net_worth_history[-1]
    return (end - start) / start


def sharpe_ratio(net_worth_history: list, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """
    Risk-adjusted return: average daily return / volatility of daily
    returns, annualized. Higher is better -- it rewards steady gains and
    penalizes wild swings, not just raw profit.
    252 = the standard number of US stock trading days per year, used
    to annualize a daily statistic for comparability across strategies.
    """
    net_worth = np.array(net_worth_history)
    daily_returns = np.diff(net_worth) / net_worth[:-1]
    excess_returns = daily_returns - (risk_free_rate / periods_per_year)

    if excess_returns.std() == 0:
        return 0.0  # no volatility -- either flat the whole time, or a single data point

    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(net_worth_history: list) -> float:
    """
    The largest peak-to-trough decline over the trajectory, as a
    fraction (e.g. -0.25 = a 25% drop from a previous high at some point).
    A key risk metric: two strategies with the same final return can have
    very different max drawdowns -- one might be a smooth climb, the
    other a terrifying rollercoaster investors would likely abandon.
    """
    net_worth = np.array(net_worth_history)
    running_max = np.maximum.accumulate(net_worth)
    drawdowns = (net_worth - running_max) / running_max
    return drawdowns.min()


if __name__ == "__main__":
    # Quick sanity test with fake data
    fake_history = [10000, 10200, 10100, 10500, 10300, 10800, 10600, 11000]
    print("Cumulative return:", cumulative_return(fake_history))
    print("Sharpe ratio:", sharpe_ratio(fake_history))
    print("Max drawdown:", max_drawdown(fake_history))