"""
data_loader.py
================
Downloads OHLCV price data and prepares it for the RL trading environment.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index: momentum oscillator from 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to a raw OHLCV dataframe."""
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_ratio"] = out["SMA_10"] / out["SMA_50"]
    out["RSI_14"] = compute_rsi(out["Close"], 14)
    out["volatility_10"] = out["log_return"].rolling(10).std()
    out = out.dropna().copy()
    return out


if __name__ == "__main__":
    data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
    featured = engineer_features(data)
    print(featured.shape)
    print(featured[["Close", "log_return", "SMA_10", "SMA_50", "SMA_ratio", "RSI_14", "volatility_10"]].head())
def chronological_split(df: pd.DataFrame, split_date: str):
    """Split strictly by date -- no shuffling, ever."""
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError("split_date leaves one side empty -- check your date range.")
    return train, test


def normalize_with_train_stats(train: pd.DataFrame, test: pd.DataFrame, cols):
    """Z-score normalize using ONLY train stats, applied to both sets."""
    means = train[cols].mean()
    stds = train[cols].std().replace(0, 1)
    train_norm = train.copy()
    test_norm = test.copy()
    for c in cols:
        train_norm[f"{c}_norm"] = (train[c] - means[c]) / stds[c]
        test_norm[f"{c}_norm"] = (test[c] - means[c]) / stds[c]
    return train_norm, test_norm, means, stds


if __name__ == "__main__":
    data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
    featured = engineer_features(data)
    train, test = chronological_split(featured, "2022-01-01")

    cols_to_normalize = ["log_return", "SMA_ratio", "RSI_14", "volatility_10"]
    train_norm, test_norm, means, stds = normalize_with_train_stats(train, test, cols_to_normalize)

    print(f"Train: {train_norm.shape}, dates {train_norm.index.min().date()} to {train_norm.index.max().date()}")
    print(f"Test:  {test_norm.shape}, dates {test_norm.index.min().date()} to {test_norm.index.max().date()}")
    print(train_norm[["log_return_norm", "SMA_ratio_norm", "RSI_14_norm", "volatility_10_norm"]].head())