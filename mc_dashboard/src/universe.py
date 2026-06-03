"""
Stock universe loader — reads master_stock_universe.csv.
Provides index-based filtering and yfinance symbol mapping.
"""

from pathlib import Path
import pandas as pd

_CSV_NAME = "master_stock_universe.csv"


def _find_csv() -> Path:
    """Walk up the directory tree to locate master_stock_universe.csv."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _CSV_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {_CSV_NAME}. "
        "Make sure it exists in the repo root."
    )


def load_universe(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else _find_csv()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["yf_symbol"] = df["Symbol"] + ".NS"   # NSE suffix for yfinance
    return df


def get_index_names(df: pd.DataFrame) -> list[str]:
    return ["All Stocks"] + sorted(df["Index Name"].unique().tolist())


def filter_by_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    if index_name == "All Stocks":
        return df
    return df[df["Index Name"] == index_name].copy()


def symbol_to_name(df: pd.DataFrame) -> dict[str, str]:
    """Map NSE symbol → company name for display."""
    return dict(zip(df["Symbol"], df["Company Name"]))
