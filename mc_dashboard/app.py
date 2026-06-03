"""
Monte Carlo Stock Price Simulator
Geometric Brownian Motion · NSE / BSE Universe
Run: streamlit run mc_dashboard/app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.universe import load_universe, get_index_names, filter_by_index, symbol_to_name
from src.data_fetcher import fetch_prices
from src.simulator import run_simulation
from src.backtester import run_backtest

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Stock Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load universe ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_universe():
    return load_universe()

universe_df = get_universe()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Simulator Settings")

    # Index filter
    index_names = get_index_names(universe_df)
    selected_index = st.selectbox("Filter by Index", index_names, index=index_names.index("Nifty 50"))

    filtered_df = filter_by_index(universe_df, selected_index)
    sym_map = symbol_to_name(filtered_df)
    sorted_symbols = sorted(sym_map.keys())

    selected_symbol = st.selectbox(
        "Select Stock",
        sorted_symbols,
        format_func=lambda s: f"{s}  —  {sym_map.get(s, '')}",
        index=sorted_symbols.index("RELIANCE") if "RELIANCE" in sorted_symbols else 0,
    )
    company_name = sym_map.get(selected_symbol, selected_symbol)

    st.divider()
    st.subheader("Calibration")
    cal_period = st.select_slider(
        "Historical Data Period",
        options=["6mo", "1y", "2y", "3y"],
        value="2y",
        help="How much history is used to estimate drift and volatility.",
    )
    cal_days_map = {"6mo": 126, "1y": 252, "2y": 504, "3y": 756}

    st.subheader("Simulation")
    horizon_label = st.select_slider(
        "Forecast Horizon",
        options=["1 Month (21d)", "2 Months (42d)", "3 Months (63d)", "6 Months (126d)", "1 Year (252d)"],
        value="3 Months (63d)",
    )
    horizon_map = {
        "1 Month (21d)": 21, "2 Months (42d)": 42, "3 Months (63d)": 63,
        "6 Months (126d)": 126, "1 Year (252d)": 252,
    }
    horizon_days = horizon_map[horizon_label]

    n_sims = st.select_slider(
        "Number of Simulations",
        options=[1000, 2000, 5000, 10000],
        value=5000,
        help="5 000 runs complete in < 1 second with NumPy vectorisation.",
    )

    use_seed = st.checkbox("Fix random seed (reproducible)", value=False)
    seed = 42 if use_seed else None

    st.divider()
    force_refresh = st.button("🔄 Refresh Price Data", use_container_width=True)


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbol: str, period: str) -> pd.Series:
    return fetch_prices(symbol, period=period)


if force_refresh:
    load_prices.clear()

with st.spinner(f"Loading {selected_symbol} prices …"):
    prices = load_prices(selected_symbol, cal_period)

if prices.empty or len(prices) < 30:
    st.error(f"Could not load sufficient price data for **{selected_symbol}**. "
             "Try a different stock or check your internet connection.")
    st.stop()

# ── run simulation ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_simulation(symbol, period, n_sims, horizon_days, seed):
    return run_simulation(prices, n_sims=n_sims, horizon_days=horizon_days, seed=seed)

t0 = time.perf_counter()
sim = compute_simulation(selected_symbol, cal_period, n_sims, horizon_days, seed)
elapsed_ms = (time.perf_counter() - t0) * 1000

percs = sim.percentile_paths()
stats = sim.summary_stats()
S0 = sim.params.last_price

# ── generate future date axis ─────────────────────────────────────────────────

last_date = prices.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)[:]   # includes day 0

# ── page header ───────────────────────────────────────────────────────────────

st.title(f"📈 {company_name} ({selected_symbol}.NS)")
st.caption(
    f"GBM Monte Carlo · {n_sims:,} simulations · {horizon_days}-day horizon · "
    f"Calibrated on {len(prices)} trading days · "
    f"Sim time: **{elapsed_ms:.0f} ms**"
)

# ── Row 1: metric cards ───────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Current Price (₹)", f"{S0:,.2f}")
c2.metric(
    "Ann. Volatility (σ)",
    f"{sim.params.sigma_annual:.1%}",
    help="Standard deviation of daily log-returns, annualised by ×√252.",
)
c3.metric(
    "Ann. Drift (μ)",
    f"{sim.params.mu_annual:.1%}",
    delta=f"{sim.params.mu_annual:.1%}",
    delta_color="normal" if sim.params.mu_annual > 0 else "inverse",
)
c4.metric(
    f"VaR 95% ({horizon_label[:8]})",
    f"{sim.var(0.95):.1%}",
    help="Maximum loss not exceeded 95% of the time at this horizon.",
)
c5.metric(
    "P(Profit)",
    f"{sim.prob_above(0):.1%}",
    help="Probability that the final price is above today's price.",
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_sim, tab_dist, tab_backtest, tab_about = st.tabs(
    ["📊 Price Fan Chart", "📉 Return Distribution", "🎯 Backtesting", "ℹ️ Methodology"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — FAN CHART
# ════════════════════════════════════════════════════════════════════════════

with tab_sim:
    col_chart, col_prob = st.columns([3, 1])

    with col_chart:
        st.subheader("Price Simulation Fan")

        # Historical tail (last 60 days for context)
        hist_tail = prices.iloc[-60:]

        fig = go.Figure()

        # Historical price line
        fig.add_scatter(
            x=hist_tail.index, y=hist_tail.values,
            name="Historical Close", mode="lines",
            line=dict(color="#AAAAAA", width=2),
        )

        # Vertical separator at last known date
        # add_vline with string dates breaks in Plotly 6.x — use add_shape instead
        fig.add_shape(
            type="line",
            x0=str(last_date)[:10], x1=str(last_date)[:10],
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="#888888", width=1),
        )
        fig.add_annotation(
            x=str(last_date)[:10], y=1, yref="paper",
            text="Today", showarrow=False,
            xanchor="right", font=dict(color="#888888", size=11),
        )

        # Fan bands — filled from outer to inner (each fill covers previous trace)
        band_defs = [
            ("p1", "p99", "rgba(96,165,250,0.08)",  "1%–99% CI"),
            ("p5", "p95", "rgba(96,165,250,0.13)",  "5%–95% CI"),
            ("p10", "p90", "rgba(96,165,250,0.18)", "10%–90% CI"),
            ("p25", "p75", "rgba(96,165,250,0.28)", "25%–75% CI"),
        ]
        for lo_col, hi_col, fill_col, label in band_defs:
            # Upper edge (invisible line needed for fill reference)
            fig.add_scatter(
                x=future_dates, y=percs[hi_col],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            )
            # Lower edge with fill back to upper
            fig.add_scatter(
                x=future_dates, y=percs[lo_col],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor=fill_col,
                name=label, hoverinfo="skip",
            )

        # Median projection
        fig.add_scatter(
            x=future_dates, y=percs["p50"],
            name="Median (50th pct)", mode="lines",
            line=dict(color="#34D399", width=2, dash="dash"),
        )

        # Current price horizontal reference
        fig.add_hline(
            y=S0, line_dash="dot", line_color="#F59E0B", line_width=1,
            annotation_text=f"₹{S0:,.0f}", annotation_position="left",
        )

        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # GBM parameters recap
        with st.expander("GBM Parameters"):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Daily Drift (μ)", f"{sim.params.mu_daily:.4f}")
            p2.metric("Daily Volatility (σ)", f"{sim.params.sigma_daily:.4f}")
            p3.metric("Calibration Obs.", f"{sim.params.n_obs:,}")
            p4.metric("Ann. Sharpe", f"{sim.params.sharpe_annual:.2f}")

    with col_prob:
        st.subheader("Outcome Probabilities")
        prob_df = sim.prob_table()
        st.dataframe(
            prob_df.style.apply(
                lambda col: [
                    "background-color:rgba(52,211,153,0.25)" if float(v.strip("%")) > 60
                    else "background-color:rgba(248,113,113,0.20)" if float(v.strip("%")) < 30
                    else ""
                    for v in col
                ] if col.name == "Probability" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            height=440,
            hide_index=True,
        )

    # Full stats table
    st.divider()
    st.subheader("Full Statistics")
    stats_display = {
        k: (f"{v:.1%}" if "P(" in k or "VaR" in k or "CVaR" in k or "Drift" in k or "Volatility" in k or "Sharpe" in k[0:3] != "Sha" and "%" in str(v) else f"₹{v:,.2f}" if "Price" in k else f"{v:,}" if isinstance(v, int) else f"{v:.4f}")
        for k, v in stats.items()
    }
    # Simpler formatting
    rows = []
    for k, v in stats.items():
        if isinstance(v, float):
            if "P(" in k or "VaR" in k or "CVaR" in k or "Drift" in k or "Vol" in k:
                formatted = f"{v:.2%}"
            elif "Sharpe" in k:
                formatted = f"{v:.2f}"
            elif "Price" in k:
                formatted = f"₹{v:,.2f}"
            else:
                formatted = f"{v:.4f}"
        else:
            formatted = str(v)
        rows.append({"Metric": k, "Value": formatted})

    stats_df = pd.DataFrame(rows)
    half = len(stats_df) // 2
    left_col, right_col = st.columns(2)
    left_col.dataframe(stats_df.iloc[:half].set_index("Metric"), use_container_width=True)
    right_col.dataframe(stats_df.iloc[half:].set_index("Metric"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RETURN DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

with tab_dist:
    st.subheader(f"Distribution of Simulated Returns at {horizon_label}")

    col_hist, col_risk = st.columns([3, 2])

    with col_hist:
        returns_pct = sim.final_returns * 100

        fig_hist = go.Figure()

        fig_hist.add_histogram(
            x=returns_pct,
            nbinsx=80,
            marker_color="#60A5FA",
            opacity=0.75,
            name="Return distribution",
        )

        # VaR lines
        var95 = sim.var(0.95) * 100
        var99 = sim.var(0.99) * 100
        cvar95 = sim.cvar(0.95) * 100

        for val, label, color in [
            (var95, f"VaR 95% ({var95:.1f}%)", "#F59E0B"),
            (var99, f"VaR 99% ({var99:.1f}%)", "#EF4444"),
            (cvar95, f"CVaR 95% ({cvar95:.1f}%)", "#EC4899"),
            (0, "Break-even", "#6EE7B7"),
        ]:
            fig_hist.add_vline(
                x=val, line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=label, annotation_position="top left",
                annotation_font_color=color,
            )

        # Shade loss region
        x_range = [float(returns_pct.min()), 0]
        fig_hist.add_vrect(
            x0=x_range[0], x1=0,
            fillcolor="rgba(239,68,68,0.06)",
            layer="below", line_width=0,
            annotation_text="Loss zone",
            annotation_position="top left",
        )

        fig_hist.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_risk:
        st.subheader("Risk Metrics")

        st.metric("VaR 95%", f"{sim.var(0.95):.2%}",
                  help="In the worst 5% of scenarios, loss exceeds this level.")
        st.metric("VaR 99%", f"{sim.var(0.99):.2%}",
                  help="In the worst 1% of scenarios, loss exceeds this level.")
        st.metric("CVaR 95%", f"{sim.cvar(0.95):.2%}",
                  help="Expected Shortfall — average loss in the worst 5% of scenarios.")

        st.divider()
        st.metric("P(> +20%)", f"{sim.prob_above(0.20):.1%}")
        st.metric("P(> +10%)", f"{sim.prob_above(0.10):.1%}")
        st.metric("P(profit)", f"{sim.prob_above(0):.1%}")
        st.metric("P(< −10%)", f"{sim.prob_below(-0.10):.1%}")
        st.metric("P(< −20%)", f"{sim.prob_below(-0.20):.1%}")

    # Cumulative distribution (fan chart in return space)
    st.divider()
    st.subheader("Cumulative Return Distribution")

    sorted_returns = np.sort(sim.final_returns * 100)
    cum_probs = np.linspace(0, 1, len(sorted_returns))

    fig_cdf = go.Figure()
    fig_cdf.add_scatter(
        x=sorted_returns, y=cum_probs * 100,
        mode="lines", line=dict(color="#60A5FA", width=2),
        name="CDF",
        hovertemplate="Return: %{x:.1f}%<br>P(≤ this return): %{y:.1f}%<extra></extra>",
    )
    fig_cdf.add_hline(y=50, line_dash="dot", line_color="#6EE7B7", annotation_text="Median", line_width=1)
    fig_cdf.add_vline(x=0, line_dash="dot", line_color="#F59E0B", annotation_text="Break-even", line_width=1)
    fig_cdf.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Return at Horizon (%)",
        yaxis_title="Cumulative Probability (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cdf, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.subheader("Rolling-Window Backtesting")
    st.caption(
        "For each window, GBM is calibrated on past data and used to forecast the next "
        f"{horizon_days} trading days. The actual price is then compared against the predicted CI."
    )

    cal_days_used = min(cal_days_map[cal_period], len(prices) - horizon_days - 5)
    if cal_days_used < 60:
        st.warning("Not enough historical data to run a meaningful backtest. Choose a longer data period.")
    else:
        with st.spinner("Running rolling backtest …"):
            @st.cache_data(show_spinner=False)
            def compute_backtest(symbol, period, horizon, cal_days):
                return run_backtest(prices, calibration_days=cal_days,
                                    horizon_days=horizon, step_days=max(3, horizon // 7))

            bt = compute_backtest(selected_symbol, cal_period, horizon_days, cal_days_used)

        if bt is None or len(bt.records) < 5:
            st.info("Too few backtest windows to display (need more historical data). "
                    "Try a longer calibration period or shorter horizon.")
        else:
            m = bt.metrics

            # ── accuracy scorecards ──────────────────────────────────────────
            st.subheader("Accuracy Scorecard")
            a1, a2, a3, a4, a5 = st.columns(5)

            def coverage_color(actual, target):
                diff = abs(actual - target)
                if diff < 0.05: return "normal"
                if diff < 0.12: return "off"
                return "inverse"

            a1.metric(
                "Coverage @80% CI", f"{m['coverage_80']:.1%}",
                delta=f"{m['coverage_80'] - 0.80:+.1%} vs target",
                delta_color=coverage_color(m["coverage_80"], 0.80),
                help="% of windows where actual price fell in the 10th–90th percentile band. Target: 80%.",
            )
            a2.metric(
                "Coverage @95% CI", f"{m['coverage_95']:.1%}",
                delta=f"{m['coverage_95'] - 0.95:+.1%} vs target",
                delta_color=coverage_color(m["coverage_95"], 0.95),
                help="% of windows where actual price fell in the 5th–95th percentile band. Target: 95%.",
            )
            a3.metric(
                "Direction Accuracy", f"{m['direction_accuracy']:.1%}",
                delta=f"{m['direction_accuracy'] - 0.50:+.1%} vs random",
                delta_color="normal" if m["direction_accuracy"] > 0.52 else "off",
                help="% of windows where the predicted direction (up/down) matched the actual direction. Baseline: 50%.",
            )
            a4.metric(
                "Median MAE", f"{m['median_mae_pct']:.2f}%",
                help="Median absolute error of the predicted median return vs actual return, as % of price.",
            )
            a5.metric(
                "Test Windows", f"{m['n_windows']:,}",
                help=f"Rolling windows of {cal_days_used}-day calibration + {horizon_days}-day forecast.",
            )

            st.divider()

            # ── rolling coverage chart ────────────────────────────────────────
            st.subheader("Predicted Confidence Bands vs Actual Price")

            df_bt = bt.records.copy()
            df_bt = df_bt.sort_values("predict_date")

            fig_bt = go.Figure()

            # 95% CI band
            fig_bt.add_scatter(
                x=df_bt["predict_date"], y=df_bt["p95"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            )
            fig_bt.add_scatter(
                x=df_bt["predict_date"], y=df_bt["p5"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor="rgba(96,165,250,0.12)",
                name="95% CI", hoverinfo="skip",
            )

            # 80% CI band
            fig_bt.add_scatter(
                x=df_bt["predict_date"], y=df_bt["p90"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            )
            fig_bt.add_scatter(
                x=df_bt["predict_date"], y=df_bt["p10"],
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor="rgba(96,165,250,0.22)",
                name="80% CI", hoverinfo="skip",
            )

            # Median forecast
            fig_bt.add_scatter(
                x=df_bt["predict_date"], y=df_bt["p50"],
                mode="lines", line=dict(color="#34D399", width=1.5, dash="dot"),
                name="Median forecast",
            )

            # Actual price — coloured by whether it's inside 80% CI
            inside_80 = df_bt[df_bt["within_80"]]
            outside_80_inside_95 = df_bt[~df_bt["within_80"] & df_bt["within_95"]]
            outside_95 = df_bt[~df_bt["within_95"]]

            for subset, color, label in [
                (inside_80,             "#34D399", "Actual (inside 80% CI)"),
                (outside_80_inside_95,  "#F59E0B", "Actual (80%–95% CI)"),
                (outside_95,            "#F87171", "Actual (outside 95% CI)"),
            ]:
                if not subset.empty:
                    fig_bt.add_scatter(
                        x=subset["actual_date"], y=subset["actual_price"],
                        mode="markers", marker=dict(color=color, size=5),
                        name=label,
                    )

            fig_bt.update_layout(
                height=380, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Forecast Start Date",
                yaxis_title="Price (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # ── calibration curve ────────────────────────────────────────────
            st.divider()
            col_calib, col_is = st.columns(2)

            with col_calib:
                st.subheader("Calibration Curve")
                st.caption("Perfect calibration = 45° line. Points above = model too conservative (bands too wide).")

                fig_calib = go.Figure()
                fig_calib.add_scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", line=dict(color="#555555", dash="dash", width=1),
                    name="Perfect calibration",
                )
                fig_calib.add_scatter(
                    x=bt.calibration["nominal"] * 100,
                    y=bt.calibration["actual"] * 100,
                    mode="lines+markers",
                    line=dict(color="#60A5FA", width=2),
                    marker=dict(size=7),
                    name=f"{selected_symbol}",
                    hovertemplate="Nominal: %{x:.0f}%<br>Actual: %{y:.1f}%<extra></extra>",
                )
                fig_calib.update_layout(
                    height=300, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="Nominal Percentile",
                    yaxis_title="Actual Fraction Below",
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(ticksuffix="%"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_calib, use_container_width=True)

            with col_is:
                st.subheader("Interval Score (IS)")
                st.caption(
                    "Proper scoring rule — lower is better. Rewards narrow CIs that still contain the outcome."
                )
                fig_is = go.Figure()
                fig_is.add_bar(
                    x=["IS @80% CI", "IS @95% CI"],
                    y=[m["mean_interval_score_80"], m["mean_interval_score_95"]],
                    marker_color=["#60A5FA", "#34D399"],
                    text=[f"{m['mean_interval_score_80']:.4f}", f"{m['mean_interval_score_95']:.4f}"],
                    textposition="outside",
                )
                fig_is.update_layout(
                    height=300, margin=dict(l=0, r=0, t=30, b=40),
                    yaxis_title="Mean Interval Score",
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_is, use_container_width=True)

            # ── raw backtest table ───────────────────────────────────────────
            with st.expander("Raw Backtest Records"):
                display_cols = ["predict_date", "actual_date", "entry_price", "actual_price",
                                "p50", "p10", "p90", "actual_return", "within_80", "within_95"]
                st.dataframe(
                    bt.records[display_cols].tail(30).style.format({
                        "entry_price": "₹{:.2f}",
                        "actual_price": "₹{:.2f}",
                        "p50": "₹{:.2f}",
                        "p10": "₹{:.2f}",
                        "p90": "₹{:.2f}",
                        "actual_return": "{:.2%}",
                    }),
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.subheader("How This Works")

    st.markdown("""
### 1 · Geometric Brownian Motion (GBM)

The price model used is **GBM** — the continuous-time model that underpins the Black-Scholes options pricing framework.

$$S(t+\\Delta t) = S(t) \\cdot \\exp\\!\\Bigl[(\\mu - \\tfrac{\\sigma^2}{2})\\,\\Delta t + \\sigma\\,\\sqrt{\\Delta t}\\,Z\\Bigr]$$

| Symbol | Meaning |
|---|---|
| $\\mu$ | Mean daily log-return (drift), estimated from historical data |
| $\\sigma$ | Std dev of daily log-returns (volatility), estimated from historical data |
| $Z$ | Standard normal random shock $Z \\sim \\mathcal{N}(0,1)$ |
| $\\Delta t$ | 1 trading day |

The $-\\sigma^2/2$ correction is the **Itô correction** — without it, the expected price would grow faster than reality because of Jensen's inequality on the exponential function.

---

### 2 · Monte Carlo

We draw all random shocks at once using NumPy:
```python
Z = np.random.default_rng(seed).standard_normal((n_sims, horizon_days))
log_increments = (mu - 0.5 * sigma**2) + sigma * Z
price_paths = S0 * np.exp(np.cumsum(log_increments, axis=1))
```
**5 000 paths × 252 days = 1.26M random numbers — vectorised in < 100 ms.**

---

### 3 · Accuracy Metrics (Backtesting)

Monte Carlo produces a *probability distribution*, not a point forecast. Accuracy therefore means **calibration**:

> *"If the model says there is a 95% chance the price falls inside a band, does the actual price land inside that band ~95% of the time?"*

| Metric | Formula | Target |
|---|---|---|
| **Coverage @80% CI** | % of windows where actual ∈ [p10, p90] | ~80% |
| **Coverage @95% CI** | % of windows where actual ∈ [p5, p95] | ~95% |
| **Direction Accuracy** | sign(actual return) == sign(median return) | > 50% |
| **Median MAE** | Median |actual − median prediction| | Lower is better |
| **Interval Score** | $(U-L) + \\frac{2}{\\alpha}(L-y)^+ + \\frac{2}{\\alpha}(y-U)^+$ | Lower is better |

The **Interval Score** is a *proper scoring rule* — it simultaneously penalises intervals that are too wide AND intervals that miss the actual outcome.

---

### 4 · Limitations

- GBM assumes **log-normal returns with constant μ and σ** — in reality, volatility clusters and tails are fat.
- The model does **not predict direction** — it reflects historical drift. Use it for *risk sizing*, not *entry signals*.
- Short calibration windows (< 1 year) may mis-estimate volatility during unusual regimes.
    """)


# ── footer ────────────────────────────────────────────────────────────────────

st.markdown(
    """<hr style="border:0.5px solid #333"/>
    <div style="text-align:center;color:#666;font-size:0.75rem">
    For educational purposes only — not financial advice.
    GBM does not predict future prices; it models probabilistic scenarios based on historical behaviour.
    </div>""",
    unsafe_allow_html=True,
)
