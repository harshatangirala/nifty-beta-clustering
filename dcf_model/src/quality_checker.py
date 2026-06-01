"""
QualityChecker — validates DCF inputs and outputs.

Each check returns a dict:
    { "status": "pass"|"warn"|"fail", "message": str, "value": ... }
"""

from datetime import datetime
from typing import Optional

import pytz

from .config import DCF_DEFAULTS, IST, FINANCIALS_STALE_DAYS

_WARN = "warn"
_PASS = "pass"
_FAIL = "fail"


class QualityChecker:

    # ── Individual checks ────────────────────────────────────────────────────

    def check_wacc(self, wacc: float) -> dict:
        lo, hi = DCF_DEFAULTS["wacc_min"], DCF_DEFAULTS["wacc_max"]
        pct = round(wacc * 100, 2)
        if wacc < lo:
            return {"status": _FAIL, "message": f"WACC {pct}% is below Indian market floor of {lo*100}%", "value": wacc}
        if wacc > hi:
            return {"status": _WARN, "message": f"WACC {pct}% is above typical Indian market ceiling of {hi*100}%", "value": wacc}
        return {"status": _PASS, "message": f"WACC {pct}% is within the 6–18% Indian market range", "value": wacc}

    def check_terminal_growth(self, terminal_g: float, wacc: float) -> dict:
        pct = round(terminal_g * 100, 2)
        max_g = DCF_DEFAULTS["perpetuity_growth_max"]
        if terminal_g <= 0:
            return {"status": _FAIL, "message": f"Terminal growth {pct}% is ≤0 — business assumed to shrink forever", "value": terminal_g}
        if terminal_g > max_g:
            return {"status": _WARN, "message": f"Terminal growth {pct}% exceeds {max_g*100}%; this implies outpacing the economy forever", "value": terminal_g}
        if terminal_g >= wacc:
            return {"status": _FAIL, "message": f"Terminal growth ({pct}%) ≥ WACC ({round(wacc*100,2)}%) — DCF formula is undefined (division by zero or negative denominator)", "value": terminal_g}
        return {"status": _PASS, "message": f"Terminal growth {pct}% is reasonable (below WACC and ≤ economy growth)", "value": terminal_g}

    def check_fcff(self, latest_fcff: Optional[float], fcff_records: list) -> dict:
        if latest_fcff is None:
            return {"status": _FAIL, "message": "Could not compute FCFF — missing income or cash-flow data", "value": None}
        cr = round(latest_fcff / 1e7, 1)
        if latest_fcff < 0:
            return {"status": _WARN, "message": f"Latest FCFF is negative (₹{cr}Cr) — projection starts from negative base; treat with caution", "value": latest_fcff}
        if len(fcff_records) >= 2:
            trend = [r["fcff"] for r in fcff_records]
            neg_count = sum(1 for v in trend if v < 0)
            if neg_count >= len(trend) // 2:
                return {"status": _WARN, "message": f"FCFF has been negative in {neg_count}/{len(trend)} reported years — operating model may be structurally FCF-negative", "value": latest_fcff}
        return {"status": _PASS, "message": f"Latest FCFF is positive at ₹{cr}Cr", "value": latest_fcff}

    def check_growth_rate(self, growth: float, stage: str = "Stage 1") -> dict:
        pct = round(growth * 100, 2)
        max_g = DCF_DEFAULTS["growth_rate_max"]
        if growth > max_g:
            return {"status": _WARN, "message": f"{stage} growth {pct}% exceeds {max_g*100}% — extremely optimistic; review carefully", "value": growth}
        if growth < -0.20:
            return {"status": _WARN, "message": f"{stage} growth {pct}% implies sustained contraction > 20% p.a.", "value": growth}
        return {"status": _PASS, "message": f"{stage} growth {pct}% is within reasonable bounds", "value": growth}

    def check_tv_weight(self, tv_pct: Optional[float]) -> dict:
        if tv_pct is None:
            return {"status": _WARN, "message": "Cannot compute terminal value weight", "value": None}
        pct = round(tv_pct * 100, 1)
        if tv_pct > 0.90:
            return {"status": _WARN, "message": f"Terminal value is {pct}% of enterprise value — model is highly sensitive to terminal assumptions", "value": tv_pct}
        if tv_pct > 0.80:
            return {"status": _WARN, "message": f"Terminal value is {pct}% of EV — normal for high-growth stocks but be cautious", "value": tv_pct}
        return {"status": _PASS, "message": f"Terminal value is {pct}% of EV — reasonable split", "value": tv_pct}

    def check_data_freshness(self, timestamp: Optional[datetime]) -> dict:
        if timestamp is None:
            return {"status": _FAIL, "message": "No financial data cached — run a refresh first", "value": None}
        now = datetime.now(IST)
        if timestamp.tzinfo is None:
            timestamp = IST.localize(timestamp)
        age_days = (now - timestamp).days
        if age_days > FINANCIALS_STALE_DAYS:
            return {"status": _FAIL, "message": f"Financial data is {age_days} days old (>{FINANCIALS_STALE_DAYS}d threshold) — refresh strongly recommended", "value": age_days}
        if age_days > 30:
            return {"status": _WARN, "message": f"Financial data is {age_days} days old — check if a new quarterly report has been released", "value": age_days}
        return {"status": _PASS, "message": f"Financial data is {age_days} day(s) old — fresh", "value": age_days}

    def check_beta(self, beta: Optional[float]) -> dict:
        if beta is None:
            return {"status": _WARN, "message": "Beta not available; defaulted to 1.0", "value": None}
        if beta < 0.3:
            return {"status": _WARN, "message": f"Beta of {beta:.2f} is unusually low — verify data source", "value": beta}
        if beta > 2.5:
            return {"status": _WARN, "message": f"Beta of {beta:.2f} is very high — high-risk stock with amplified market sensitivity", "value": beta}
        return {"status": _PASS, "message": f"Beta of {beta:.2f} is within normal range", "value": beta}

    # ── Full report ──────────────────────────────────────────────────────────

    def full_report(
        self,
        dcf_result: dict,
        fin_timestamp: Optional[datetime] = None,
    ) -> dict:
        """
        Run all checks given a completed DCF result dict.
        Returns dict of {check_name: {status, message, value}}.
        """
        if not dcf_result.get("valid"):
            return {
                "data_error": {
                    "status": _FAIL,
                    "message": dcf_result.get("error", "DCF did not produce a valid result"),
                    "value": None,
                }
            }

        wacc = dcf_result.get("wacc", 0)
        tg = dcf_result.get("terminal_growth", 0)
        fcff_records = dcf_result.get("fcff_history", [])
        latest_fcff = dcf_result.get("base_fcff")
        beta = dcf_result.get("wacc_breakdown", {}).get("beta")
        tv_pct = dcf_result.get("tv_pct_of_ev")
        g1 = dcf_result.get("growth_stage1", 0)
        g2 = dcf_result.get("growth_stage2", 0)

        checks = {
            "wacc_range":       self.check_wacc(wacc),
            "terminal_growth":  self.check_terminal_growth(tg, wacc),
            "fcff_quality":     self.check_fcff(latest_fcff, fcff_records),
            "stage1_growth":    self.check_growth_rate(g1, "Stage 1"),
            "stage2_growth":    self.check_growth_rate(g2, "Stage 2"),
            "tv_weight":        self.check_tv_weight(tv_pct),
            "data_freshness":   self.check_data_freshness(fin_timestamp),
            "beta_sanity":      self.check_beta(beta),
        }

        # Overall summary
        statuses = [c["status"] for c in checks.values()]
        if _FAIL in statuses:
            overall = _FAIL
        elif _WARN in statuses:
            overall = _WARN
        else:
            overall = _PASS

        checks["_overall"] = {
            "status": overall,
            "message": f"{statuses.count(_PASS)} pass / {statuses.count(_WARN)} warn / {statuses.count(_FAIL)} fail",
            "value": None,
        }
        return checks
