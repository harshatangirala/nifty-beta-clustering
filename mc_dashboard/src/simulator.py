"""
GBM Monte Carlo Simulator
=========================

Model: Geometric Brownian Motion (GBM) — the standard continuous-time model for equity prices.

    S(t + Δt) = S(t) · exp( (μ - σ²/2)·Δt + σ·√Δt·Z )

where
    μ  = mean daily log-return (drift, estimated from history)
    σ  = std dev of daily log-returns (volatility, estimated from history)
    Z  ~ N(0,1) independent random shocks
    Δt = 1 day

All n_sims paths are generated with a single vectorised NumPy call, so
5 000 simulations × 252 steps runs in ~50 ms.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── parameter estimation ─────────────────────────────────────────────────────

@dataclass
class GBMParams:
    mu_daily: float         # mean daily log-return
    sigma_daily: float      # std dev of daily log-returns
    last_price: float
    n_obs: int              # number of calibration observations

    @property
    def mu_annual(self) -> float:
        return self.mu_daily * 252

    @property
    def sigma_annual(self) -> float:
        return self.sigma_daily * np.sqrt(252)

    @property
    def sharpe_annual(self) -> float:
        rf_daily = 0.07 / 252
        return (self.mu_daily - rf_daily) / (self.sigma_daily + 1e-12) * np.sqrt(252)


def calibrate(prices: pd.Series) -> GBMParams:
    """Estimate μ and σ from historical closing prices using log-returns."""
    if len(prices) < 10:
        raise ValueError("Need at least 10 price observations to calibrate GBM.")
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return GBMParams(
        mu_daily=float(log_ret.mean()),
        sigma_daily=float(log_ret.std(ddof=1)),
        last_price=float(prices.iloc[-1]),
        n_obs=len(log_ret),
    )


# ── simulation ───────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    paths: np.ndarray          # shape (n_sims, horizon + 1)
    params: GBMParams
    horizon: int
    n_sims: int
    seed: Optional[int]

    # Derived properties (computed on first access)
    _final: Optional[np.ndarray] = field(default=None, repr=False)
    _returns: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def final_prices(self) -> np.ndarray:
        if self._final is None:
            self._final = self.paths[:, -1]
        return self._final

    @property
    def final_returns(self) -> np.ndarray:
        if self._returns is None:
            self._returns = (self.final_prices / self.params.last_price) - 1
        return self._returns

    # ── percentile bands ─────────────────────────────────────────────────────

    def percentile_paths(self, levels: list[int] | None = None) -> pd.DataFrame:
        """
        Return percentile price paths across all simulations.
        Columns: p5, p10, p25, p50, p75, p90, p95 by default.
        Rows: time steps 0 … horizon.
        """
        if levels is None:
            levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        arr = np.percentile(self.paths, levels, axis=0).T   # (horizon+1, n_levels)
        return pd.DataFrame(arr, columns=[f"p{l}" for l in levels])

    # ── summary statistics ────────────────────────────────────────────────────

    def var(self, confidence: float = 0.95) -> float:
        """VaR: the return at the (1-confidence) lower tail."""
        return float(np.percentile(self.final_returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """CVaR (Expected Shortfall): mean return in the tail beyond VaR."""
        var_level = self.var(confidence)
        tail = self.final_returns[self.final_returns <= var_level]
        return float(tail.mean()) if len(tail) > 0 else var_level

    def prob_above(self, threshold_return: float = 0.0) -> float:
        """P(final_return > threshold_return)."""
        return float((self.final_returns > threshold_return).mean())

    def prob_below(self, threshold_return: float) -> float:
        return float((self.final_returns < threshold_return).mean())

    def summary_stats(self) -> dict:
        S0 = self.params.last_price
        r = self.final_returns
        return {
            "Current Price": S0,
            "Median Final Price": float(np.median(self.final_prices)),
            "Mean Final Price": float(np.mean(self.final_prices)),
            "Annualised Drift (μ)": self.params.mu_annual,
            "Annualised Volatility (σ)": self.params.sigma_annual,
            "Sharpe Ratio (ann.)": self.params.sharpe_annual,
            "VaR 95% (horizon)": self.var(0.95),
            "VaR 99% (horizon)": self.var(0.99),
            "CVaR 95% (horizon)": self.cvar(0.95),
            "P(profit)": self.prob_above(0.0),
            "P(>+10%)": self.prob_above(0.10),
            "P(>+20%)": self.prob_above(0.20),
            "P(<-10%)": self.prob_below(-0.10),
            "P(<-20%)": self.prob_below(-0.20),
            "Simulations": self.n_sims,
            "Horizon (days)": self.horizon,
            "Calibration obs.": self.params.n_obs,
        }

    def prob_table(self, thresholds_pct: list[float] | None = None) -> pd.DataFrame:
        """Return a table of P(return > threshold) for display."""
        if thresholds_pct is None:
            thresholds_pct = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 50]
        rows = []
        for t in thresholds_pct:
            rows.append({
                "Return Threshold": f"{t:+.0f}%",
                "Probability": f"{self.prob_above(t / 100):.1%}",
                "Price Target (₹)": f"{self.params.last_price * (1 + t / 100):,.1f}",
            })
        return pd.DataFrame(rows)


# ── main entry point ──────────────────────────────────────────────────────────

def run_simulation(
    prices: pd.Series,
    n_sims: int = 5000,
    horizon_days: int = 21,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Calibrate GBM on historical prices and run Monte Carlo simulation.

    Parameters
    ----------
    prices       : historical closing prices (chronological order)
    n_sims       : number of simulation paths (5 000 completes in <100 ms)
    horizon_days : number of trading days to project forward
    seed         : optional RNG seed for reproducibility
    """
    params = calibrate(prices)
    rng = np.random.default_rng(seed)

    # Generate all random shocks at once — vectorised GBM
    Z = rng.standard_normal((n_sims, horizon_days))                  # (N, T)
    log_increments = (params.mu_daily - 0.5 * params.sigma_daily**2) + params.sigma_daily * Z
    log_paths = np.cumsum(log_increments, axis=1)                    # (N, T)

    # Prepend S(0) and exponentiate
    price_paths = params.last_price * np.exp(
        np.hstack([np.zeros((n_sims, 1)), log_paths])                # (N, T+1)
    )

    return SimulationResult(
        paths=price_paths,
        params=params,
        horizon=horizon_days,
        n_sims=n_sims,
        seed=seed,
    )
