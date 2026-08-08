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
    # yfinance sometimes returns a MultiIndex column even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    data = download_price_data("AAPL", "2015-01-01", "2024-01-01")
    print(data.shape)
    print(data.head())