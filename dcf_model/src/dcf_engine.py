"""
DCF Engine — core valuation logic.

Calculation chain
-----------------
  1. FCFF  = EBIT × (1 – tax_rate) + D&A + ΔWorking Capital + Capex
             (Capex and ΔWC are signed as they appear in the cash-flow statement;
              both are typically negative, reducing FCFF)
  2. WACC  = (E/V) × Ke  +  (D/V) × Kd × (1 – tax_rate)
             Ke  = Rf + β × ERP   (CAPM)
             Kd  = |Interest Expense| / Total Debt
  3. DCF   = Σ  FCFF_t / (1+WACC)^t   +   TV / (1+WACC)^n
             TV  = FCFF_n × (1 + g) / (WACC – g)
  4. Intrinsic value per share = DCF / Shares Outstanding
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import DCF_DEFAULTS, INDIA_EQUITY_RISK_PREMIUM, INDIA_RISK_FREE_RATE

log = logging.getLogger(__name__)


def _get(field_map: dict, candidates: list[str], default=None):
    """
    Pull the most-recent non-null value for the first matching field name.

    field_map structure coming from DataFetcher:
        { "YYYY-MM-DDTHH:MM:SS": { "Field Name": float|None, ... }, ... }
    """
    if not field_map:
        return default
    dates = sorted(field_map.keys(), reverse=True)
    for field in candidates:
        for date in dates:
            v = field_map[date].get(field)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
    return default


def _series_values(field_map: dict, candidates: list[str]) -> list[float]:
    """Return a time-ordered list of values for a field (oldest first)."""
    if not field_map:
        return []
    dates = sorted(field_map.keys())  # ascending
    values = []
    for field in candidates:
        vals = []
        for date in dates:
            v = field_map[date].get(field)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if vals:
            values = vals
            break
    return values


class DCFEngine:
    def __init__(
        self,
        risk_free_rate: float = INDIA_RISK_FREE_RATE,
        equity_risk_premium: float = INDIA_EQUITY_RISK_PREMIUM,
    ):
        self.rf = risk_free_rate
        self.erp = equity_risk_premium

    # ── FCFF ────────────────────────────────────────────────────────────────

    def calculate_fcff(self, fin_data: dict) -> dict:
        """
        Return dict with annual FCFF values and component breakdown.
        Tries annual first; falls back to summing last 4 quarters.
        """
        income = fin_data.get("annual_income", {})
        cashflow = fin_data.get("annual_cashflow", {})

        # Pull series (oldest → newest) for trend
        dates = sorted(set(list(income.keys()) + list(cashflow.keys())))[-4:]  # last 4 years

        records = []
        for date in dates:
            inc = income.get(date, {})
            cf = cashflow.get(date, {})

            ebit = (
                inc.get("Operating Income")
                or inc.get("EBIT")
                or inc.get("Normalized EBITDA")  # rough fallback
            )
            tax_provision = inc.get("Tax Provision") or inc.get("Income Tax Expense")
            pretax = inc.get("Pretax Income") or inc.get("Income Before Tax") or 1

            dna = (
                cf.get("Depreciation And Amortization")
                or cf.get("Reconciled Depreciation")
                or cf.get("Depreciation")
                or 0
            )
            delta_wc = cf.get("Change In Working Capital") or 0
            capex = cf.get("Capital Expenditure") or cf.get("Purchase Of PPE") or 0

            if ebit is None:
                continue

            tax_rate = abs(float(tax_provision or 0)) / max(abs(float(pretax)), 1)
            tax_rate = min(tax_rate, 0.40)  # cap at 40% (Indian corporate tax ~25–30%)

            ebit = float(ebit)
            dna = float(dna)
            delta_wc = float(delta_wc)
            capex = float(capex)

            # FCFF = EBIT(1-t) + D&A + ΔWC + Capex
            # Note: capex < 0 in yfinance (cash outflow), ΔWC sign varies
            nopat = ebit * (1 - tax_rate)
            fcff = nopat + dna + delta_wc + capex

            records.append({
                "date": date[:10],
                "ebit": ebit,
                "tax_rate": tax_rate,
                "nopat": nopat,
                "dna": dna,
                "delta_wc": delta_wc,
                "capex": capex,
                "fcff": fcff,
            })

        return {"records": records, "latest_fcff": records[-1]["fcff"] if records else None}

    # ── WACC ────────────────────────────────────────────────────────────────

    def calculate_wacc(self, fin_data: dict) -> dict:
        info = fin_data.get("info", {})
        income = fin_data.get("annual_income", {})
        balance = fin_data.get("annual_balance", {})

        market_cap = float(info.get("marketCap") or 0)
        total_debt = _get(balance, ["Total Debt", "Long Term Debt", "Total Long Term Debt"]) or 0

        # Enterprise structure
        total_value = market_cap + total_debt
        if total_value == 0:
            total_value = 1  # avoid zero-div; WACC will be unreliable
        we = market_cap / total_value
        wd = total_debt / total_value

        # Cost of equity (CAPM)
        beta = float(info.get("beta") or 1.0)
        beta = max(0.3, min(beta, 3.0))  # clamp extreme betas
        ke = self.rf + beta * self.erp

        # Cost of debt
        interest = abs(_get(income, ["Interest Expense", "Net Interest Income", "Interest Income"]) or 0)
        kd_raw = interest / max(total_debt, 1)
        kd = kd_raw if 0.01 < kd_raw < 0.25 else self.rf + 0.02  # fallback: Rf + 2%

        # Tax rate
        tax_provision = abs(_get(income, ["Tax Provision", "Income Tax Expense"]) or 0)
        pretax = abs(_get(income, ["Pretax Income", "Income Before Tax"]) or 1)
        tax_rate = min(tax_provision / max(pretax, 1), 0.40)

        wacc = we * ke + wd * kd * (1 - tax_rate)

        return {
            "wacc": wacc,
            "cost_of_equity": ke,
            "cost_of_debt": kd,
            "weight_equity": we,
            "weight_debt": wd,
            "beta": beta,
            "risk_free_rate": self.rf,
            "equity_risk_premium": self.erp,
            "tax_rate": tax_rate,
            "market_cap": market_cap,
            "total_debt": total_debt,
        }

    # ── Growth rate estimation ───────────────────────────────────────────────

    def estimate_growth_rate(self, fin_data: dict) -> float:
        """Estimate historical revenue CAGR from last 3–4 years of annual data."""
        revenue_series = _series_values(
            fin_data.get("annual_income", {}),
            ["Total Revenue", "Revenue"],
        )
        if len(revenue_series) < 2:
            return 0.10  # default 10% if no history

        revenue_series = [v for v in revenue_series if v and v > 0]
        if len(revenue_series) < 2:
            return 0.10

        years = len(revenue_series) - 1
        cagr = (revenue_series[-1] / revenue_series[0]) ** (1 / years) - 1
        return max(-0.20, min(cagr, 0.50))  # clamp to [-20%, 50%]

    # ── DCF projection ──────────────────────────────────────────────────────

    def project_fcff(
        self,
        base_fcff: float,
        growth_stage1: float,
        growth_stage2: float,
        terminal_growth: float,
        projection_years: int = 5,
        stage1_years: int = 3,
    ) -> list[dict]:
        """
        Project FCFF for each year using a 2-stage growth model that declines to terminal.
        Returns list of {year, growth_rate, fcff}.
        """
        projections = []
        fcff = base_fcff
        for yr in range(1, projection_years + 1):
            g = growth_stage1 if yr <= stage1_years else growth_stage2
            fcff = fcff * (1 + g)
            projections.append({"year": yr, "growth_rate": g, "fcff": fcff})
        return projections

    # ── Full DCF valuation ───────────────────────────────────────────────────

    def run_dcf(
        self,
        fin_data: dict,
        wacc: Optional[float] = None,
        growth_stage1: Optional[float] = None,
        growth_stage2: Optional[float] = None,
        terminal_growth: float = DCF_DEFAULTS["terminal_growth_rate"],
        projection_years: int = DCF_DEFAULTS["projection_years"],
        stage1_years: int = DCF_DEFAULTS["growth_stage1_years"],
    ) -> dict:
        """
        Run full DCF and return valuation dict.

        Parameters
        ----------
        fin_data        : output of DataFetcher.get_financials()
        wacc            : override computed WACC (0–1 float)
        growth_stage1/2 : override stage growth rates
        """
        # ── FCFF ──
        fcff_result = self.calculate_fcff(fin_data)
        latest_fcff = fcff_result.get("latest_fcff")
        if latest_fcff is None or latest_fcff == 0:
            return {"error": "Insufficient FCFF data", "valid": False}

        # ── WACC ──
        wacc_result = self.calculate_wacc(fin_data)
        effective_wacc = wacc if wacc is not None else wacc_result["wacc"]
        effective_wacc = max(0.01, effective_wacc)  # must be positive

        # ── Growth rates ──
        hist_growth = self.estimate_growth_rate(fin_data)
        g1 = growth_stage1 if growth_stage1 is not None else hist_growth
        g2 = growth_stage2 if growth_stage2 is not None else max(terminal_growth, hist_growth * 0.6)

        # ── Project FCFF ──
        projections = self.project_fcff(
            base_fcff=latest_fcff,
            growth_stage1=g1,
            growth_stage2=g2,
            terminal_growth=terminal_growth,
            projection_years=projection_years,
            stage1_years=stage1_years,
        )

        # ── Present values ──
        pv_fcff = []
        for item in projections:
            pv = item["fcff"] / (1 + effective_wacc) ** item["year"]
            pv_fcff.append({**item, "pv": pv})

        # ── Terminal value ──
        final_fcff = projections[-1]["fcff"]
        if effective_wacc <= terminal_growth:
            return {"error": "WACC must exceed terminal growth rate", "valid": False}
        tv = final_fcff * (1 + terminal_growth) / (effective_wacc - terminal_growth)
        pv_tv = tv / (1 + effective_wacc) ** projection_years

        # ── Enterprise value ──
        sum_pv_fcff = sum(p["pv"] for p in pv_fcff)
        enterprise_value = sum_pv_fcff + pv_tv

        # ── Equity value ──
        total_debt = wacc_result.get("total_debt", 0)
        cash = _get(fin_data.get("annual_balance", {}), ["Cash And Cash Equivalents", "Cash"]) or 0
        equity_value = enterprise_value - total_debt + cash

        # ── Per-share ──
        shares = float(fin_data.get("info", {}).get("sharesOutstanding") or 0)
        intrinsic_per_share = equity_value / shares if shares > 0 else None

        return {
            "valid": True,
            "intrinsic_value_per_share": intrinsic_per_share,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "sum_pv_fcff": sum_pv_fcff,
            "pv_terminal_value": pv_tv,
            "terminal_value": tv,
            "tv_pct_of_ev": pv_tv / enterprise_value if enterprise_value else None,
            "wacc": effective_wacc,
            "wacc_computed": wacc_result["wacc"],
            "wacc_breakdown": wacc_result,
            "growth_stage1": g1,
            "growth_stage2": g2,
            "terminal_growth": terminal_growth,
            "base_fcff": latest_fcff,
            "projections": pv_fcff,
            "fcff_history": fcff_result["records"],
            "hist_growth": hist_growth,
            "shares_outstanding": shares,
            "cash": cash,
            "total_debt": total_debt,
        }

    # ── Sensitivity analysis ─────────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        fin_data: dict,
        base_wacc: float,
        base_terminal_growth: float,
        current_price: float,
        wacc_step: float = 0.005,
        g_step: float = 0.005,
        wacc_range: int = 4,
        g_range: int = 2,
    ) -> pd.DataFrame:
        """
        2-D sensitivity table of intrinsic value vs WACC and terminal growth rate.
        Returns DataFrame with WACC as index, terminal_g as columns.
        Values show intrinsic value per share.
        """
        wacc_values = [round(base_wacc + i * wacc_step, 4) for i in range(-wacc_range, wacc_range + 1)]
        g_values = [round(base_terminal_growth + j * g_step, 4) for j in range(-g_range, g_range + 1)]

        table = {}
        for g in g_values:
            col = {}
            for w in wacc_values:
                if w <= g or w <= 0:
                    col[w] = None
                    continue
                result = self.run_dcf(fin_data, wacc=w, terminal_growth=g)
                col[w] = result.get("intrinsic_value_per_share")
            table[g] = col

        df = pd.DataFrame(table)
        df.index.name = "WACC"
        df.columns.name = "Terminal Growth"
        return df

    # ── Revenue/FCFF trend series ────────────────────────────────────────────

    def get_financial_summary(self, fin_data: dict) -> pd.DataFrame:
        """Return a clean DataFrame with annual Revenue, EBIT, FCFF for display."""
        fcff_result = self.calculate_fcff(fin_data)
        rows = fcff_result.get("records", [])
        if not rows:
            return pd.DataFrame()

        income = fin_data.get("annual_income", {})
        records = []
        for r in rows:
            date = r["date"]
            rev = None
            for d in income.keys():
                if d.startswith(date[:7]):  # match YYYY-MM
                    rev = income[d].get("Total Revenue") or income[d].get("Revenue")
                    break
            records.append({
                "Year": date[:4],
                "Revenue (₹Cr)": round((rev or 0) / 1e7, 1),
                "EBIT (₹Cr)": round(r["ebit"] / 1e7, 1),
                "FCFF (₹Cr)": round(r["fcff"] / 1e7, 1),
                "Tax Rate (%)": round(r["tax_rate"] * 100, 1),
                "EBIT Margin (%)": round(r["ebit"] / max(rev or 1, 1) * 100, 1),
            })
        return pd.DataFrame(records)
