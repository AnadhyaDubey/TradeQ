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