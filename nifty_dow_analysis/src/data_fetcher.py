"""Fetches Nifty 50 daily OHLC history from yfinance and caches it to disk."""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "nifty50_ohlc.csv"

TICKER = "^NSEI"
START_DATE = "2015-01-01"
END_DATE = "2026-06-17"  # yfinance end is exclusive, so this captures 16-Jun-2026


def fetch_nifty_ohlc(force_refresh: bool = False) -> pd.DataFrame:
    """Returns a DataFrame indexed by trading date with Open/High/Low/Close/Volume columns."""
    if not force_refresh and CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.to_csv(CACHE_FILE)
    return df


if __name__ == "__main__":
    data = fetch_nifty_ohlc(force_refresh=True)
    print(f"Fetched {len(data)} rows from {data.index.min().date()} to {data.index.max().date()}")
    print(data.tail())
