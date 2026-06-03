"""
Rolling-window backtester for the GBM Monte Carlo model.

What "accuracy" means for a probabilistic model
------------------------------------------------
Monte Carlo does NOT predict a single future price — it produces a
probability distribution. Accuracy therefore means *calibration*: does
the stated confidence interval (CI) actually contain the true outcome
at the stated frequency?

Metrics computed
----------------
1.  Coverage @80% CI  — % of test windows where actual price fell inside
                         [10th pct, 90th pct] of simulated distribution.
                         Well-calibrated model → ~80%.  Too wide → >80%.

2.  Coverage @95% CI  — Same for [5th, 95th] pct.  Target: ~95%.

3.  Direction accuracy — % of windows where sign(actual return) ==
                         sign(median simulated return).  Baseline: 50%.

4.  Median MAE (%)    — Mean |actual_return − median_return| across windows.
                         Measures central-tendency error regardless of spread.

5.  Interval Score    — Proper scoring rule (Winkler, 1972):
                         IS(α) = (U − L) + (2/α)·max(0, L−y) + (2/α)·max(0, y−U)
                         where [L, U] = CI at level (1−α), y = actual price.
                         Lower IS is better; it rewards narrow intervals that
                         still contain the outcome, and penalises misses heavily.

6.  Calibration curve — For each decile, what fraction of actual outcomes
                         fell below that decile of the simulated distribution?
                         Perfect calibration → straight 45-degree line.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .simulator import run_simulation


@dataclass
class BacktestResult:
    records: pd.DataFrame       # one row per test window
    metrics: dict               # summary scalars
    calibration: pd.DataFrame   # decile calibration curve


def _interval_score(lower: float, upper: float, actual: float, alpha: float) -> float:
    """Winkler interval score — lower is better."""
    penalty = (2 / alpha) * (max(0.0, lower - actual) + max(0.0, actual - upper))
    return (upper - lower) + penalty


def run_backtest(
    prices: pd.Series,
    calibration_days: int = 252,
    horizon_days: int = 21,
    step_days: int = 5,
    n_sims: int = 500,        # fewer sims here — speed matters
    seed: int = 42,
) -> Optional[BacktestResult]:
    """
    Rolling-window backtest.

    For each window:
      - Calibrate GBM on [start : start + calibration_days]
      - Simulate horizon_days forward
      - Compare against actual price at [start + calibration_days + horizon_days]
    """
    n = len(prices)
    min_needed = calibration_days + horizon_days + 1
    if n < min_needed:
        return None

    records = []
    rng = np.random.default_rng(seed)
    local_seed = int(rng.integers(0, 2**31))

    for start in range(0, n - min_needed, step_days):
        cal_end = start + calibration_days
        test_idx = cal_end + horizon_days

        if test_idx >= n:
            break

        cal_prices = prices.iloc[start:cal_end]
        actual_price = float(prices.iloc[test_idx])
        entry_price = float(cal_prices.iloc[-1])

        try:
            sim = run_simulation(cal_prices, n_sims=n_sims, horizon_days=horizon_days, seed=local_seed)
            local_seed = (local_seed + 1) % (2**31)
        except Exception:
            continue

        final = sim.final_prices
        percs = {p: float(np.percentile(final, p)) for p in [5, 10, 25, 50, 75, 90, 95]}

        actual_ret = (actual_price - entry_price) / entry_price
        median_ret = (percs[50] - entry_price) / entry_price

        within_80 = percs[10] <= actual_price <= percs[90]
        within_95 = percs[5] <= actual_price <= percs[95]

        # Rank of actual price in simulated distribution (for calibration curve)
        empirical_rank = float((final <= actual_price).mean())

        # Interval scores (normalised by entry price for comparability across stocks)
        is_80 = _interval_score(percs[10] / entry_price, percs[90] / entry_price,
                                 actual_ret, alpha=0.20)
        is_95 = _interval_score(percs[5] / entry_price, percs[95] / entry_price,
                                 actual_ret, alpha=0.05)

        records.append({
            "window_start": prices.index[start],
            "predict_date": prices.index[cal_end],
            "actual_date": prices.index[test_idx],
            "entry_price": entry_price,
            "actual_price": actual_price,
            "actual_return": actual_ret,
            "p5": percs[5],
            "p10": percs[10],
            "p25": percs[25],
            "p50": percs[50],
            "p75": percs[75],
            "p90": percs[90],
            "p95": percs[95],
            "median_return": median_ret,
            "within_80": within_80,
            "within_95": within_95,
            "correct_direction": (actual_ret > 0) == (median_ret > 0),
            "empirical_rank": empirical_rank,
            "interval_score_80": is_80,
            "interval_score_95": is_95,
            "abs_error_return": abs(actual_ret - median_ret),
        })

    if not records:
        return None

    df = pd.DataFrame(records)

    # ── summary metrics ─────────────────────────────────────────────────────
    metrics = {
        "n_windows": len(df),
        "coverage_80": float(df["within_80"].mean()),
        "coverage_95": float(df["within_95"].mean()),
        "direction_accuracy": float(df["correct_direction"].mean()),
        "median_mae_pct": float(df["abs_error_return"].median() * 100),
        "mean_interval_score_80": float(df["interval_score_80"].mean()),
        "mean_interval_score_95": float(df["interval_score_95"].mean()),
        "calibration_days": calibration_days,
        "horizon_days": horizon_days,
    }

    # ── calibration curve ─────────────────────────────────────────────────
    deciles = np.arange(0, 1.01, 0.10)
    actual_fracs = [float((df["empirical_rank"] <= d).mean()) for d in deciles]
    calib = pd.DataFrame({"nominal": deciles, "actual": actual_fracs})

    return BacktestResult(records=df, metrics=metrics, calibration=calib)
