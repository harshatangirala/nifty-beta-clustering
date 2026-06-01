"""
DataUpdateScheduler — APScheduler-based automatic refresh.

Schedule
--------
  Daily   @ 16:00 IST (Mon–Fri)   → refresh closing prices for all Nifty 50
  Weekly  @ 08:00 IST on Monday   → check financials; refresh if >30 days stale
  Monthly (1st of earnings months: Jan, Apr, Jul, Oct) → force-refresh all financials

The scheduler is designed to run as a Streamlit @st.cache_resource singleton
so that it survives page re-runs without spawning duplicate threads.
"""

import logging
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import pytz

from .config import EARNINGS_MONTHS, IST, NIFTY50_STOCKS

log = logging.getLogger(__name__)

_ALL_SYMBOLS = list(NIFTY50_STOCKS.keys())


class DataUpdateScheduler:
    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.scheduler = BackgroundScheduler(timezone=IST)
        self._last_run: dict[str, Optional[datetime]] = {
            "price_refresh": None,
            "weekly_financials": None,
            "quarterly_financials": None,
        }
        self._status_log: list[dict] = []
        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        try:
            self.scheduler.add_job(
                self._daily_price_update,
                CronTrigger(day_of_week="mon-fri", hour=16, minute=0, timezone=IST),
                id="daily_price",
                replace_existing=True,
            )
            self.scheduler.add_job(
                self._weekly_financials_update,
                CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=IST),
                id="weekly_financials",
                replace_existing=True,
            )
            self.scheduler.add_job(
                self._quarterly_full_refresh,
                CronTrigger(month=",".join(map(str, EARNINGS_MONTHS)), day=1, hour=9, minute=0, timezone=IST),
                id="quarterly_financials",
                replace_existing=True,
            )
            self.scheduler.start()
            self._running = True
            log.info("DataUpdateScheduler started")
        except Exception as exc:
            # Background threads may be restricted on Streamlit Cloud — degrade gracefully
            log.warning("Scheduler could not start (background threads may be disabled): %s", exc)
            self._running = False

    def stop(self):
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ── scheduled jobs ───────────────────────────────────────────────────────

    def _daily_price_update(self):
        log.info("[Scheduler] Daily price refresh started")
        results = self.fetcher.refresh_all(_ALL_SYMBOLS, data_type="price")
        ok = sum(1 for v in results.values() if v and v.get("price") is not None)
        self._record_run("price_refresh", f"Refreshed {ok}/{len(_ALL_SYMBOLS)} prices")

    def _weekly_financials_update(self):
        log.info("[Scheduler] Weekly financials check started")
        stale = []
        for sym in _ALL_SYMBOLS:
            ts = self.fetcher.get_cache_timestamp(sym, "financials")
            if ts is None:
                stale.append(sym)
                continue
            age_days = (datetime.now(IST) - ts).days
            if age_days > 30:
                stale.append(sym)

        if stale:
            log.info("[Scheduler] Refreshing financials for %d stale symbols", len(stale))
            self.fetcher.refresh_all(stale, data_type="financials")

        self._record_run("weekly_financials", f"Checked {len(_ALL_SYMBOLS)}; refreshed {len(stale)}")

    def _quarterly_full_refresh(self):
        log.info("[Scheduler] Quarterly full refresh started")
        self.fetcher.refresh_all(_ALL_SYMBOLS, data_type="financials")
        self._record_run("quarterly_financials", f"Force-refreshed all {len(_ALL_SYMBOLS)} financials")

    # ── manual trigger ───────────────────────────────────────────────────────

    def force_refresh(self, symbol: str) -> dict:
        log.info("[Scheduler] Manual refresh for %s", symbol)
        price = self.fetcher.get_current_price(symbol, force_refresh=True)
        fin = self.fetcher.get_financials(symbol, force_refresh=True)
        self._record_run("manual", f"Manual refresh: {symbol}")
        return {"price": price, "financials": fin}

    # ── internal helpers ─────────────────────────────────────────────────────

    def _record_run(self, job: str, message: str):
        now = datetime.now(IST)
        self._last_run[job] = now
        self._status_log.append({"job": job, "time": now.isoformat(), "message": message})
        if len(self._status_log) > 100:
            self._status_log = self._status_log[-100:]

    def get_next_run_times(self) -> dict:
        if not self._running:
            return {}
        result = {}
        for job in self.scheduler.get_jobs():
            next_fire = job.next_run_time
            result[job.id] = next_fire.isoformat() if next_fire else None
        return result

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "last_runs": {k: (v.isoformat() if v else None) for k, v in self._last_run.items()},
            "next_runs": self.get_next_run_times(),
            "recent_log": self._status_log[-10:],
        }
