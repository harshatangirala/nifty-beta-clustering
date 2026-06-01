"""
Nifty 50 DCF Valuation Dashboard
Run: streamlit run dcf_model/app.py
"""

import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable when launched from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DCF_DEFAULTS, NIFTY50_STOCKS, IST
from src.data_fetcher import DataFetcher
from src.dcf_engine import DCFEngine
from src.quality_checker import QualityChecker
from src.scheduler import DataUpdateScheduler

# ─── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nifty 50 DCF Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── shared singletons (survive Streamlit reruns) ─────────────────────────────

@st.cache_resource
def init_components():
    fetcher = DataFetcher()
    engine = DCFEngine()
    checker = QualityChecker()
    scheduler = DataUpdateScheduler(fetcher)
    try:
        scheduler.start()
    except Exception:
        pass  # APScheduler may fail in some cloud environments; non-critical
    return fetcher, engine, checker, scheduler


fetcher, engine, checker, scheduler = init_components()

# ─── helpers ──────────────────────────────────────────────────────────────────

STATUS_COLOR = {"pass": "🟢", "warn": "🟡", "fail": "🔴"}

def freshness_badge(ts_str: str | None, stale_days: int = 1) -> str:
    if not ts_str:
        return "🔴 No data"
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        age = (datetime.now(IST) - ts).total_seconds() / 3600
        age_label = f"{int(age)}h ago" if age < 48 else f"{int(age/24)}d ago"
        color = "🟢" if age < stale_days * 24 else ("🟡" if age < stale_days * 72 else "🔴")
        return f"{color} {age_label}"
    except Exception:
        return "⚪ Unknown"


def fmt_cr(value: float | None) -> str:
    if value is None:
        return "N/A"
    cr = value / 1e7
    if abs(cr) >= 1e5:
        return f"₹{cr/1e5:.2f}L Cr"
    if abs(cr) >= 1e3:
        return f"₹{cr/1e3:.1f}K Cr"
    return f"₹{cr:.1f} Cr"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%" if value is not None else "N/A"


def fmt_inr(value: float | None) -> str:
    return f"₹{value:,.1f}" if value is not None else "N/A"


# ─── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 DCF Settings")

    selected = st.selectbox(
        "Select Nifty 50 Stock",
        options=list(NIFTY50_STOCKS.keys()),
        format_func=lambda s: f"{s} – {NIFTY50_STOCKS[s]}",
        index=list(NIFTY50_STOCKS.keys()).index("RELIANCE"),
    )

    force_refresh_btn = st.button("🔄 Force Refresh Data", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Macro Assumptions")
    rf_rate = st.slider("Risk-Free Rate (Rf %)", 4.0, 10.0, 7.0, 0.1) / 100
    erp_val = st.slider("Equity Risk Premium (ERP %)", 4.0, 12.0, 7.0, 0.1) / 100
    terminal_g = st.slider("Terminal Growth Rate (g %)", 1.0, 7.0, 4.0, 0.1) / 100
    proj_years = st.select_slider("Projection Years", options=[5, 7, 10], value=5)

    st.divider()
    st.subheader("Growth Assumptions")
    g1_pct = st.slider("Stage 1 Growth (Yrs 1–3 %)", -10.0, 50.0, 15.0, 0.5) / 100
    g2_pct = st.slider("Stage 2 Growth (Yrs 4–5 %)", -10.0, 30.0, 8.0, 0.5) / 100

    st.divider()
    override_wacc = st.checkbox("Override computed WACC")
    if override_wacc:
        custom_wacc = st.slider("Custom WACC (%)", 6.0, 22.0, 12.0, 0.25) / 100
    else:
        custom_wacc = None

    st.divider()
    st.caption("🕐 Scheduler Status")
    sched_status = scheduler.get_status()
    st.caption(f"Running: {'✅' if sched_status['running'] else '❌'}")
    next_runs = sched_status.get("next_runs", {})
    if next_runs.get("daily_price"):
        st.caption(f"Next price update: {next_runs['daily_price'][:16]}")


# ─── main content ──────────────────────────────────────────────────────────────

st.title(f"📊 {NIFTY50_STOCKS.get(selected, selected)} ({selected}) — DCF Valuation")
st.caption(f"All monetary values in Indian Rupees · IST timestamps · Data: yfinance + NSE")

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def load_price(symbol: str) -> dict:
    return fetcher.get_current_price(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def load_financials(symbol: str) -> dict:
    return fetcher.get_financials(symbol)


# Trigger refresh if button was pressed
if force_refresh_btn:
    with st.spinner(f"Refreshing data for {selected} …"):
        load_price.clear()
        load_financials.clear()
        result = scheduler.force_refresh(selected)
    st.success("Data refreshed successfully!")

price_data = load_price(selected)
fin_data = load_financials(selected)

# ── rebuild engine with current sidebar macro values ─────────────────────────
live_engine = DCFEngine(risk_free_rate=rf_rate, equity_risk_premium=erp_val)

# Run DCF with sidebar parameters
dcf = live_engine.run_dcf(
    fin_data,
    wacc=custom_wacc,
    growth_stage1=g1_pct,
    growth_stage2=g2_pct,
    terminal_growth=terminal_g,
    projection_years=proj_years,
    stage1_years=min(3, proj_years),
)

current_price = price_data.get("price")
intrinsic = dcf.get("intrinsic_value_per_share")
upside = ((intrinsic / current_price) - 1) if (intrinsic and current_price) else None

# ─── Row 1: key metric cards ─────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    source_label = "Live LTP" if price_data.get("source") == "live_ltp" else "Prev Close"
    badge = freshness_badge(price_data.get("timestamp"), stale_days=1)
    st.metric(
        label=f"Market Price ({source_label})",
        value=fmt_inr(current_price),
        delta=badge,
        delta_color="off",
    )

with col2:
    if intrinsic:
        delta_str = f"{upside:+.1%}" if upside is not None else "N/A"
        delta_col = "normal" if upside and upside > 0 else "inverse"
        st.metric("Intrinsic Value (DCF)", fmt_inr(intrinsic), delta=delta_str, delta_color=delta_col)
    else:
        st.metric("Intrinsic Value (DCF)", "N/A")

with col3:
    wacc_val = dcf.get("wacc")
    ke = dcf.get("wacc_breakdown", {}).get("cost_of_equity")
    kd = dcf.get("wacc_breakdown", {}).get("cost_of_debt")
    st.metric("WACC", fmt_pct(wacc_val))
    st.caption(f"Ke: {fmt_pct(ke)} · Kd: {fmt_pct(kd)}")

with col4:
    info = fin_data.get("info", {})
    pe = info.get("trailingPE") or info.get("forwardPE")
    mktcap = info.get("marketCap")
    st.metric("P/E Ratio", f"{pe:.1f}x" if pe else "N/A")
    st.caption(f"Market Cap: {fmt_cr(mktcap)}")

st.divider()

# ─── Row 2: DCF breakdown + gauge ────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("FCF Projections & Present Values")
    if dcf.get("valid") and dcf.get("projections"):
        proj_df = pd.DataFrame(dcf["projections"])
        proj_df["FCFF (₹Cr)"] = proj_df["fcff"] / 1e7
        proj_df["PV of FCFF (₹Cr)"] = proj_df["pv"] / 1e7
        proj_df["Year"] = proj_df["year"].astype(str)
        proj_df["Growth (%)"] = (proj_df["growth_rate"] * 100).round(1)

        fig = go.Figure()
        fig.add_bar(x=proj_df["Year"], y=proj_df["FCFF (₹Cr)"],
                    name="Projected FCFF", marker_color="#4C9BE8")
        fig.add_bar(x=proj_df["Year"], y=proj_df["PV of FCFF (₹Cr)"],
                    name="PV of FCFF", marker_color="#2ECC71")
        pv_tv_cr = dcf.get("pv_terminal_value", 0) / 1e7
        fig.add_bar(x=[f"TV (Yr{proj_years})"], y=[pv_tv_cr],
                    name="PV of Terminal Value", marker_color="#E67E22")
        fig.update_layout(
            barmode="group", height=320, margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="₹ Crore",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary row
        tv_pct = dcf.get("tv_pct_of_ev", 0) or 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Enterprise Value", fmt_cr(dcf.get("enterprise_value")))
        c2.metric("Equity Value", fmt_cr(dcf.get("equity_value")))
        c3.metric("Terminal Value Weight", fmt_pct(tv_pct))
    else:
        err = dcf.get("error", "DCF could not be computed — check data availability")
        st.warning(f"⚠️ {err}")

with col_right:
    st.subheader("Intrinsic vs Market Price")
    if intrinsic and current_price:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_price,
            delta={"reference": intrinsic, "valueformat": ".0f",
                   "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            title={"text": f"Market Price (₹)<br><span style='font-size:0.7em'>vs Intrinsic ₹{intrinsic:,.0f}</span>"},
            gauge={
                "axis": {"range": [0, max(current_price, intrinsic) * 1.5]},
                "bar": {"color": "#4C9BE8"},
                "threshold": {
                    "line": {"color": "green", "width": 4},
                    "thickness": 0.75,
                    "value": intrinsic,
                },
                "steps": [
                    {"range": [0, intrinsic * 0.8], "color": "#d4efdf"},
                    {"range": [intrinsic * 0.8, intrinsic * 1.0], "color": "#fef9e7"},
                    {"range": [intrinsic * 1.0, max(current_price, intrinsic) * 1.5], "color": "#fadbd8"},
                ],
            },
            number={"prefix": "₹", "valueformat": ",.0f"},
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        margin_of_safety = (intrinsic - current_price) / intrinsic if intrinsic else 0
        verdict = (
            "🟢 Undervalued" if margin_of_safety > 0.20 else
            "🟡 Fairly Valued" if margin_of_safety > -0.10 else
            "🔴 Overvalued"
        )
        st.markdown(f"**Verdict:** {verdict}  \n**Margin of Safety:** {fmt_pct(margin_of_safety)}")
    else:
        st.info("Run a valuation to see the gauge.")

st.divider()

# ─── Row 3: sensitivity analysis ─────────────────────────────────────────────

st.subheader("Sensitivity Analysis — Intrinsic Value per Share (₹)")
st.caption("Vary WACC and terminal growth to see how the valuation changes.")

if dcf.get("valid") and dcf.get("wacc") and fin_data.get("annual_income"):
    with st.spinner("Computing sensitivity grid …"):
        base_wacc_for_sens = dcf["wacc"]
        sens_df = live_engine.sensitivity_analysis(
            fin_data,
            base_wacc=base_wacc_for_sens,
            base_terminal_growth=terminal_g,
            current_price=current_price or 0,
        )

    if not sens_df.empty:
        # Format column / index labels as %
        sens_df.columns = [f"{c*100:.1f}%" for c in sens_df.columns]
        sens_df.index = [f"{i*100:.1f}%" for i in sens_df.index]
        sens_df.index.name = "WACC ↓  /  Terminal g →"

        # Style: green = intrinsic > current price, red = intrinsic < current price
        def style_cell(val):
            if val is None or pd.isna(val):
                return "background-color: #f0f0f0; color: #aaa"
            if current_price and val > current_price * 1.1:
                intensity = min(int((val / current_price - 1) * 50), 120)
                return f"background-color: rgba(46,204,113,{0.3 + intensity/400}); font-weight: bold"
            if current_price and val < current_price * 0.9:
                intensity = min(int((1 - val / current_price) * 50), 120)
                return f"background-color: rgba(231,76,60,{0.3 + intensity/400}); font-weight: bold"
            return "background-color: rgba(241,196,15,0.3)"

        styled = (
            sens_df.style
            .map(style_cell)
            .format(lambda v: f"₹{v:,.0f}" if v is not None and not pd.isna(v) else "–")
        )
        st.dataframe(styled, use_container_width=True, height=280)
        st.caption(f"🟢 Green = intrinsic > market price by >10% · 🟡 Yellow = ≈ fair value · 🔴 Red = overvalued by >10%")
else:
    st.info("Sensitivity analysis requires a successful DCF run.")

st.divider()

# ─── Row 4: historical financials ────────────────────────────────────────────

col_hist, col_wacc = st.columns([3, 2])

with col_hist:
    st.subheader("Historical Financials")
    fin_summary = live_engine.get_financial_summary(fin_data)
    if not fin_summary.empty:
        st.dataframe(fin_summary.set_index("Year"), use_container_width=True)

        # Revenue + FCFF trend chart
        fig_trend = go.Figure()
        fig_trend.add_scatter(
            x=fin_summary["Year"], y=fin_summary["Revenue (₹Cr)"],
            name="Revenue", mode="lines+markers", line=dict(color="#4C9BE8", width=2),
        )
        fig_trend.add_scatter(
            x=fin_summary["Year"], y=fin_summary["FCFF (₹Cr)"],
            name="FCFF", mode="lines+markers", line=dict(color="#2ECC71", width=2),
        )
        fig_trend.add_scatter(
            x=fin_summary["Year"], y=fin_summary["EBIT (₹Cr)"],
            name="EBIT", mode="lines+markers", line=dict(color="#E67E22", width=2, dash="dot"),
        )
        fig_trend.update_layout(
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h"),
            yaxis_title="₹ Crore",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No historical financial data available.")

with col_wacc:
    st.subheader("WACC Breakdown")
    if dcf.get("valid"):
        wb = dcf.get("wacc_breakdown", {})
        wacc_data = {
            "Component": ["Risk-Free Rate (Rf)", "Equity Risk Premium (ERP)", "Beta (β)",
                          "Cost of Equity (Ke)", "Cost of Debt (Kd)", "Tax Rate",
                          "Weight of Equity (E/V)", "Weight of Debt (D/V)", "WACC"],
            "Value": [
                fmt_pct(wb.get("risk_free_rate")),
                fmt_pct(wb.get("equity_risk_premium")),
                f"{wb.get('beta', 0):.2f}",
                fmt_pct(wb.get("cost_of_equity")),
                fmt_pct(wb.get("cost_of_debt")),
                fmt_pct(wb.get("tax_rate")),
                fmt_pct(wb.get("weight_equity")),
                fmt_pct(wb.get("weight_debt")),
                fmt_pct(dcf.get("wacc")),
            ],
        }
        df_wacc = pd.DataFrame(wacc_data)
        st.dataframe(df_wacc.set_index("Component"), use_container_width=True, height=280)

        # Waterfall: Ke contribution vs Kd contribution
        e_contrib = wb.get("weight_equity", 0) * wb.get("cost_of_equity", 0)
        d_contrib = wb.get("weight_debt", 0) * wb.get("cost_of_debt", 0) * (1 - wb.get("tax_rate", 0))
        fig_wf = go.Figure(go.Pie(
            labels=["Equity Cost Contribution", "Debt Cost Contribution (after-tax)"],
            values=[e_contrib, d_contrib],
            hole=0.5,
            marker_colors=["#4C9BE8", "#E67E22"],
        ))
        fig_wf.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0),
                             showlegend=True, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# ─── Row 5: quality checks ────────────────────────────────────────────────────

st.subheader("Quality Checks")

fin_ts = fetcher.get_cache_timestamp(selected, "financials")
qc_report = checker.full_report(dcf, fin_timestamp=fin_ts)

overall = qc_report.pop("_overall", None)
if overall:
    color = {"pass": "success", "warn": "warning", "fail": "error"}[overall["status"]]
    getattr(st, color)(f"**Overall Quality:** {STATUS_COLOR[overall['status']]} {overall['message']}")

qc_cols = st.columns(4)
for i, (check_name, result) in enumerate(qc_report.items()):
    col = qc_cols[i % 4]
    icon = STATUS_COLOR[result["status"]]
    label = check_name.replace("_", " ").title()
    col.markdown(f"**{icon} {label}**")
    col.caption(result["message"])

st.divider()

# ─── Row 6: data freshness & scheduler log ────────────────────────────────────

st.subheader("Data Freshness")
cf1, cf2, cf3 = st.columns(3)

with cf1:
    st.markdown("**Price Data**")
    st.markdown(freshness_badge(price_data.get("timestamp"), stale_days=1))
    st.caption(f"Source: {price_data.get('source', 'unknown')}")
    st.caption(f"Last fetched: {(price_data.get('timestamp') or 'N/A')[:19]}")

with cf2:
    st.markdown("**Financial Data**")
    fin_ts_str = fin_data.get("timestamp")
    st.markdown(freshness_badge(fin_ts_str, stale_days=30))
    st.caption(f"Latest report date: {fin_data.get('report_date', 'N/A')}")
    st.caption(f"Last fetched: {(fin_ts_str or 'N/A')[:19]}")

with cf3:
    st.markdown("**Scheduler**")
    last_runs = sched_status.get("last_runs", {})
    st.caption(f"Last price refresh: {(last_runs.get('price_refresh') or 'Not yet run')[:19]}")
    st.caption(f"Last financial check: {(last_runs.get('weekly_financials') or 'Not yet run')[:19]}")
    if next_runs:
        for job, t in next_runs.items():
            st.caption(f"Next {job}: {(t or 'N/A')[:16]}")

# ─── expandable: raw DCF assumptions recap ────────────────────────────────────

with st.expander("DCF Assumptions Summary"):
    assump = {
        "Stock": f"{selected} – {NIFTY50_STOCKS.get(selected)}",
        "Risk-Free Rate": fmt_pct(rf_rate),
        "Equity Risk Premium": fmt_pct(erp_val),
        "Beta": f"{dcf.get('wacc_breakdown', {}).get('beta', 'N/A'):.2f}" if dcf.get("valid") else "N/A",
        "Cost of Equity (Ke)": fmt_pct(dcf.get("wacc_breakdown", {}).get("cost_of_equity")),
        "Cost of Debt (Kd)": fmt_pct(dcf.get("wacc_breakdown", {}).get("cost_of_debt")),
        "WACC": fmt_pct(dcf.get("wacc")),
        "Stage 1 Growth (Yrs 1–3)": fmt_pct(g1_pct),
        "Stage 2 Growth (Yrs 4–5)": fmt_pct(g2_pct),
        "Terminal Growth Rate": fmt_pct(terminal_g),
        "Projection Years": proj_years,
        "Base FCFF": fmt_cr(dcf.get("base_fcff")),
        "Historical Revenue CAGR": fmt_pct(dcf.get("hist_growth")),
        "Shares Outstanding": f"{(dcf.get('shares_outstanding') or 0)/1e7:.2f} Cr shares",
        "Cash & Equivalents": fmt_cr(dcf.get("cash")),
        "Total Debt": fmt_cr(dcf.get("total_debt")),
    }
    st.table(pd.DataFrame(list(assump.items()), columns=["Parameter", "Value"]).set_index("Parameter"))

# ─── footer ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style="border:0.5px solid #444"/>
    <div style="text-align:center; color:#888; font-size:0.75rem">
    This dashboard is for educational and research purposes only. Not financial advice.
    DCF valuations are sensitive to assumptions — always apply a margin of safety.
    </div>
    """,
    unsafe_allow_html=True,
)
