import re
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np
import warnings

warnings.filterwarnings('ignore')
# Project-level data folder
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def clean_download(ticker: str, start: str) -> pd.DataFrame:
    """Download daily OHLC prices for `ticker` from `start` and save them to data/<name>.csv."""
    data = yf.download(tickers=ticker, start=start, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker or your connection.")

    # Recent yfinance versions return (field, ticker) column pairs even for one ticker
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)

    data = data[["Open", "High", "Low", "Close"]]
    data.index.name = "Date"

    # File-safe name
    name = re.sub(r"[^A-Za-z0-9]", "", ticker)

    DATA_DIR.mkdir(exist_ok=True)
    data.to_csv(DATA_DIR / f"{name}.csv")

    return data


def compute_losses(prices: pd.DataFrame) -> pd.Series:
    """Returns the loss as the negative of the returns, as a Series named "loss", with the first (empty) row dropped."""

    returns = np.log(prices["Close"]).diff()
    loss = -returns

    return loss.dropna().rename('loss')


def parkinson_variance(prices: pd.DataFrame) -> pd.Series:
    """Returns the Parkinson daily ranged-based proxy for realized variance as a Series."""

    highoverlow = np.log(prices['High'] / prices['Low'])
    parkinson_var = (1 / (4 * np.log(2))) * highoverlow ** 2

    return parkinson_var.rename("parkinson_var")

def check_prices(prices: pd.DataFrame) -> dict:
    """Verifies a given DataFrame for missing, non-positive, High<Low, Closes outside the daily range, stale quotes, duplicate dates, or unsorted index"""

    info_dict = {}

    info_dict['missing'] = prices.isna().sum().sum()
    info_dict['non_positive'] = (prices[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    info_dict['high_below_low'] = (prices['High'] < prices['Low']).sum()
    info_dict['close_outside_range'] = ((prices['Close'] > prices['High']) | (prices['Close'] < prices['Low'])).sum()
    info_dict['zero_range'] = (prices['High'] == prices['Low']).sum()
    info_dict['duplicate_dates'] = prices.index.duplicated().sum()
    info_dict['unsorted'] = 0 if prices.index.is_monotonic_increasing else 1

    return info_dict


prices = clean_download("^GSPC", "2015-01-01")
loss = compute_losses(prices)
pv = parkinson_variance(prices)
print(check_prices(prices))

