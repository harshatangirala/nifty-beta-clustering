"""
Price data fetcher — wraps yfinance with a lightweight JSON file cache.
Cache TTL: 6 hours (intraday), refreshed on force_refresh or expiry.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _cache_root() -> Path:
    preferred = Path(__file__).resolve().parents[1] / "data" / "cache"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        (preferred / ".probe").touch()
        (preferred / ".probe").unlink()
        return preferred
    except (PermissionError, OSError):
        fallback = Path(tempfile.gettempdir()) / "mc_cache"
        fallback.mkdir(exist_ok=True)
        return fallback


_CACHE = _cache_root()
_TTL_HOURS = 6


class PriceCache:
    """Simple disk-backed cache for closing-price time series."""

    def _path(self, key: str) -> Path:
        return _CACHE / f"{key}.json"

    def get(self, key: str) -> Optional[pd.Series]:
        p = self._path(key)
        if not p.exists():
            return None
        age_h = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
        if age_h > _TTL_HOURS:
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return pd.Series(
                raw["prices"],
                index=pd.to_datetime(raw["dates"], utc=True).tz_convert(None),
                dtype=float,
                name="Close",
            )
        except Exception:
            return None

    def put(self, key: str, series: pd.Series):
        data = {
            "prices": [float(v) for v in series.values],
            "dates": [str(d)[:10] for d in series.index],
            "saved": datetime.now().isoformat(),
        }
        try:
            self._path(key).write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            log.debug("Cache write failed: %s", exc)


_disk_cache = PriceCache()


def fetch_prices(
    symbol: str,
    period: str = "3y",
    force_refresh: bool = False,
) -> pd.Series:
    """
    Return daily closing prices for an NSE symbol (e.g. 'RELIANCE').
    Always appends '.NS' for yfinance.
    """
    key = f"{symbol}_{period}"
    if not force_refresh:
        cached = _disk_cache.get(key)
        if cached is not None and len(cached) > 10:
            return cached

    yf_sym = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period=period, auto_adjust=True)
        if hist.empty:
            log.warning("No data returned for %s", yf_sym)
            return pd.Series(dtype=float)
        series = hist["Close"].dropna()
        # Strip timezone so downstream code is tz-naive
        series.index = series.index.tz_localize(None) if series.index.tz is not None else series.index
        _disk_cache.put(key, series)
        return series
    except Exception as exc:
        log.warning("yfinance fetch failed for %s: %s", yf_sym, exc)
        return pd.Series(dtype=float)


def fetch_current_price(symbol: str) -> Optional[float]:
    try:
        t = yf.Ticker(f"{symbol}.NS")
        fi = t.fast_info
        return getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
    except Exception:
        return None
