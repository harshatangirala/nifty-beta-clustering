"""
DataFetcher: yfinance wrapper with per-symbol JSON cache and freshness logic.

Price freshness rules
---------------------
  Live hours (9:15–15:30 IST, Mon–Fri) → LTP from fast_info, re-fetch if >5 min stale
  After hours / weekend                → previous close, re-fetch once per calendar day

Financials freshness rule
-------------------------
  Re-fetch if cache is >90 days old OR force_refresh=True
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytz
import yfinance as yf

from .config import (
    IST,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    PRICE_STALE_AFTERHOURS_DAYS,
    PRICE_STALE_SECONDS,
    FINANCIALS_STALE_DAYS,
)

log = logging.getLogger(__name__)

def _resolve_cache_root() -> Path:
    """Use app-local cache dir; fall back to /tmp if not writable (e.g. Streamlit Cloud)."""
    preferred = Path(__file__).resolve().parents[1] / "data" / "cache"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.touch()
        probe.unlink()
        return preferred
    except (PermissionError, OSError):
        fallback = Path(tempfile.gettempdir()) / "dcf_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

_CACHE_ROOT = _resolve_cache_root()


# ─── helpers ────────────────────────────────────────────────────────────────

def _now_ist() -> datetime:
    return datetime.now(IST)


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string back to aware datetime."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt


def _df_to_serialisable(df: pd.DataFrame) -> dict:
    """Convert financial DataFrame (dates as columns) to JSON-serialisable dict."""
    if df is None or df.empty:
        return {}
    out = {}
    for col in df.columns:
        key = col.isoformat() if hasattr(col, "isoformat") else str(col)
        out[key] = {
            str(idx): (None if pd.isna(val) else float(val))
            for idx, val in df[col].items()
        }
    return out


def _latest_value(field_map: dict, candidates: list[str]) -> Optional[float]:
    """Return the most-recent non-null value from a list of candidate field names."""
    if not field_map:
        return None
    # field_map structure: { "YYYY-MM-DD": { "Field Name": value, ... }, ... }
    dates = sorted(field_map.keys(), reverse=True)
    for field in candidates:
        for date in dates:
            v = field_map[date].get(field)
            if v is not None and not (isinstance(v, float) and v != v):
                return v
    return None


# ─── DataFetcher ────────────────────────────────────────────────────────────

class DataFetcher:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_ROOT
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── market timing ───────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        now = _now_ist()
        if now.weekday() >= 5:      # Saturday = 5, Sunday = 6
            return False
        open_dt = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
        close_dt = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)
        return open_dt <= now <= close_dt

    # ── price fetching ──────────────────────────────────────────────────────

    def get_current_price(self, symbol: str, force_refresh: bool = False) -> dict:
        cache_key = f"{symbol}_price"
        cached = self._load_cache(cache_key)

        if not force_refresh and cached:
            ts = _parse_ts(cached["timestamp"])
            age_s = (_now_ist() - ts).total_seconds()
            if self.is_market_open() and age_s < PRICE_STALE_SECONDS:
                return cached
            if not self.is_market_open() and age_s < PRICE_STALE_AFTERHOURS_DAYS * 86400:
                return cached

        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            if self.is_market_open():
                fi = ticker.fast_info
                price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
                source = "live_ltp"
            else:
                hist = ticker.history(period="2d")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                source = "previous_close"

            result = {
                "symbol": symbol,
                "price": price,
                "source": source,
                "market_open": self.is_market_open(),
                "timestamp": _now_ist().isoformat(),
            }
            self._save_cache(cache_key, result)
            return result

        except Exception as exc:
            log.warning("Price fetch failed for %s: %s", symbol, exc)
            if cached:
                cached["_stale_fallback"] = True
                return cached
            return {"symbol": symbol, "price": None, "source": "error", "timestamp": _now_ist().isoformat()}

    # ── financial data fetching ─────────────────────────────────────────────

    def get_financials(self, symbol: str, force_refresh: bool = False) -> dict:
        cache_key = f"{symbol}_financials"
        cached = self._load_cache(cache_key)

        if not force_refresh and cached:
            ts = _parse_ts(cached["timestamp"])
            age_days = (_now_ist() - ts).days
            if age_days < FINANCIALS_STALE_DAYS:
                return cached

        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = {}
            try:
                raw_info = ticker.info or {}
                # Keep only JSON-serialisable scalar fields
                info = {k: v for k, v in raw_info.items() if isinstance(v, (int, float, str, bool, type(None)))}
            except Exception:
                pass

            result = {
                "symbol": symbol,
                "info": info,
                "annual_income":    _df_to_serialisable(self._safe_df(ticker, "financials")),
                "annual_balance":   _df_to_serialisable(self._safe_df(ticker, "balance_sheet")),
                "annual_cashflow":  _df_to_serialisable(self._safe_df(ticker, "cashflow")),
                "qtr_income":       _df_to_serialisable(self._safe_df(ticker, "quarterly_financials")),
                "qtr_balance":      _df_to_serialisable(self._safe_df(ticker, "quarterly_balance_sheet")),
                "qtr_cashflow":     _df_to_serialisable(self._safe_df(ticker, "quarterly_cashflow")),
                "timestamp": _now_ist().isoformat(),
                "report_date": self._latest_report_date(ticker),
            }
            self._save_cache(cache_key, result)
            return result

        except Exception as exc:
            log.warning("Financials fetch failed for %s: %s", symbol, exc)
            if cached:
                cached["_stale_fallback"] = True
                return cached
            return {"symbol": symbol, "info": {}, "timestamp": _now_ist().isoformat()}

    # ── historical prices for trend analysis ───────────────────────────────

    def get_historical_prices(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=period)
            return df[["Open", "High", "Low", "Close", "Volume"]] if not df.empty else pd.DataFrame()
        except Exception as exc:
            log.warning("Historical price fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    # ── cache helpers ───────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, data: dict):
        p = self._cache_path(key)
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        except Exception as exc:
            log.warning("Cache write failed for %s: %s", key, exc)

    def get_cache_timestamp(self, symbol: str, data_type: str = "financials") -> Optional[datetime]:
        cached = self._load_cache(f"{symbol}_{data_type}")
        if cached and "timestamp" in cached:
            return _parse_ts(cached["timestamp"])
        return None

    # ── yfinance helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe_df(ticker: yf.Ticker, attr: str) -> pd.DataFrame:
        try:
            df = getattr(ticker, attr, None)
            return df if (df is not None and not df.empty) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _latest_report_date(ticker: yf.Ticker) -> Optional[str]:
        try:
            df = ticker.quarterly_financials
            if df is not None and not df.empty:
                return str(max(df.columns))
        except Exception:
            pass
        return None

    # ── batch helper ────────────────────────────────────────────────────────

    def refresh_all(self, symbols: list[str], data_type: str = "price"):
        """Batch refresh for scheduler use."""
        results = {}
        for sym in symbols:
            try:
                if data_type == "price":
                    results[sym] = self.get_current_price(sym, force_refresh=True)
                else:
                    results[sym] = self.get_financials(sym, force_refresh=True)
            except Exception as exc:
                log.warning("Batch refresh failed for %s: %s", sym, exc)
                results[sym] = None
        return results
