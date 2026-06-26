"""
Nifty 50 — Seasonality Study
Layer 1: which day of the week historically performs best/worst.
Layer 2: which month, week-of-month, and week-of-year historically perform best/worst.
Run: streamlit run nifty_dow_analysis/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher import fetch_nifty_ohlc, TICKER
from src.analysis import compute_returns, weekday_summary, yearly_weekday_means, overall_stats, build_report, WEEKDAY_ORDER
from src.seasonality import (
    month_summary, week_of_month_summary, week_of_year_summary, yearly_month_means,
    build_seasonality_report, MONTH_ORDER, WEEK_OF_MONTH_ORDER,
)

st.set_page_config(page_title="Nifty 50 Seasonality Study", page_icon="📅", layout="wide")

WEEKDAY_COLORS = {
    "Monday": "#378ADD",
    "Tuesday": "#BA7517",
    "Wednesday": "#639922",
    "Thursday": "#7F77DD",
    "Friday": "#993556",
}


@st.cache_data(ttl=3600)
def load_data(force_refresh: bool = False):
    ohlc = fetch_nifty_ohlc(force_refresh=force_refresh)
    df = compute_returns(ohlc)
    return df


def styled_bar(data, x, y, color_map=None, pct_suffix="%", hline=None, height=None):
    fig = px.bar(data, x=x, y=y, color=x, text=y, color_discrete_map=color_map)
    fig.update_traces(texttemplate=f"%{{text:.3f}}{pct_suffix}", textposition="outside")
    if hline is not None:
        fig.add_hline(y=hline, line_width=1, line_color="gray", line_dash="dash" if hline != 0 else "solid")
    fig.update_layout(
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12), height=height,
    )
    return fig


st.title("📅 Nifty 50 — Seasonality Study")
st.caption(
    f"Daily % change vs. previous close, {TICKER} (Nifty 50), 1 Jan 2015 – 16 Jun 2026. "
    "Source: Yahoo Finance via yfinance."
)

with st.sidebar:
    st.header("Data")
    if st.button("🔄 Refresh data from Yahoo Finance"):
        load_data.clear()
        st.session_state["_force_refresh"] = True
    st.caption("Data is cached locally in `data/nifty50_ohlc.csv` after the first fetch.")

force = st.session_state.pop("_force_refresh", False)
df_full = load_data(force_refresh=force)

years = sorted(df_full["Year"].unique())
with st.sidebar:
    st.header("Filters")
    year_range = st.select_slider(
        "Year range", options=years, value=(years[0], years[-1])
    )
df = df_full[(df_full["Year"] >= year_range[0]) & (df_full["Year"] <= year_range[1])]

stats = overall_stats(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trading days analysed", f"{stats['total_trading_days']:,}")
c2.metric("Date range", f"{stats['start_date']} → {stats['end_date']}")
c3.metric("Overall avg. daily return", f"{stats['overall_mean_return']:+.3f}%")
c4.metric("Overall win rate", f"{stats['overall_win_rate']:.1f}%")

st.divider()

tab_dow, tab_season = st.tabs(["📆 Layer 1 — Day of Week", "🗓️ Layer 2 — Seasonality (Month & Week)"])

# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DAY OF WEEK
# ════════════════════════════════════════════════════════════════════════════
with tab_dow:
    summary = weekday_summary(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average % return by weekday")
        bar_df = summary.reset_index()
        st.plotly_chart(
            styled_bar(bar_df, "Weekday", "mean_return", WEEKDAY_COLORS, hline=0),
            width="stretch",
        )
    with col2:
        st.subheader("Win rate by weekday")
        fig = px.bar(bar_df, x="Weekday", y="win_rate_pct", color="Weekday",
                     color_discrete_map=WEEKDAY_COLORS, text="win_rate_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Distribution of daily returns")
        fig = px.box(df, x="Weekday", y="PctChange", color="Weekday",
                     color_discrete_map=WEEKDAY_COLORS, category_orders={"Weekday": WEEKDAY_ORDER}, points=False)
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12), yaxis_range=[-4, 4])
        st.plotly_chart(fig, width="stretch")
    with col4:
        st.subheader("Compounded return if you only held that weekday")
        cum_df = summary.reset_index()
        st.plotly_chart(
            styled_bar(cum_df, "Weekday", "cumulative_return_pct", WEEKDAY_COLORS, hline=0),
            width="stretch",
        )

    st.subheader("Year-by-year: average return per weekday")
    heat = yearly_weekday_means(df)
    fig = go.Figure(data=go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index.astype(str),
        colorscale="RdYlGn", zmid=0, text=heat.values, texttemplate="%{text:.2f}%",
        colorbar=dict(title="Avg %"),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12), height=350)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.header("📋 Summary report — Day of Week")
    st.markdown(build_report(summary))

    st.subheader("Full weekday statistics")
    display_summary = summary.rename(columns={
        "trading_days": "Trading Days", "mean_return": "Mean Return %", "median_return": "Median Return %",
        "std_dev": "Std Dev %", "best_day": "Best Day %", "worst_day": "Worst Day %",
        "win_rate_pct": "Win Rate %", "cumulative_return_pct": "Cumulative Return %",
    })
    st.dataframe(display_summary, width="stretch")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SEASONALITY (MONTH & WEEK)
# ════════════════════════════════════════════════════════════════════════════
with tab_season:
    month_sum = month_summary(df)
    wom_sum = week_of_month_summary(df)
    woy_sum = week_of_year_summary(df)

    st.subheader("Average % return by month")
    month_bar = month_sum.reset_index().rename(columns={"index": "MonthName"})
    fig = px.bar(month_bar, x="MonthName", y="mean_return", color="MonthName",
                 category_orders={"MonthName": MONTH_ORDER},
                 color_discrete_sequence=px.colors.qualitative.Set3, text="mean_return")
    fig.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win rate by month")
        fig = px.bar(month_bar, x="MonthName", y="win_rate_pct", color="MonthName",
                     category_orders={"MonthName": MONTH_ORDER},
                     color_discrete_sequence=px.colors.qualitative.Set3, text="win_rate_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Compounded return if you only held that month")
        fig = px.bar(month_bar, x="MonthName", y="cumulative_return_pct", color="MonthName",
                     category_orders={"MonthName": MONTH_ORDER},
                     color_discrete_sequence=px.colors.qualitative.Set3, text="cumulative_return_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Year-by-year: average return per month")
    heat = yearly_month_means(df)
    fig = go.Figure(data=go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index.astype(str),
        colorscale="RdYlGn", zmid=0, text=heat.values, texttemplate="%{text:.2f}%",
        colorbar=dict(title="Avg %"),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12), height=420)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Turn-of-month effect: average return by week-in-month")
        wom_bar = wom_sum.reset_index().rename(columns={"index": "WeekOfMonth"})
        fig = px.bar(wom_bar, x="WeekOfMonth", y="mean_return", color="WeekOfMonth",
                     category_orders={"WeekOfMonth": WEEK_OF_MONTH_ORDER},
                     color_discrete_sequence=px.colors.qualitative.Pastel, text="mean_return")
        fig.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Week 1 = days 1-7 of the month, Week 2 = days 8-14, ... Week 5 = days 29-31 (fewer observations).")

    with col4:
        st.subheader("Win rate by week-in-month")
        fig = px.bar(wom_bar, x="WeekOfMonth", y="win_rate_pct", color="WeekOfMonth",
                     category_orders={"WeekOfMonth": WEEK_OF_MONTH_ORDER},
                     color_discrete_sequence=px.colors.qualitative.Pastel, text="win_rate_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Week-of-year effect: average return by ISO week number")
    woy_bar = woy_sum.reset_index()
    fig = px.bar(woy_bar, x="ISOWeek", y="mean_return", text=None,
                 color="mean_return", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12),
                       xaxis=dict(dtick=2, title="ISO week number (1-53)"), yaxis_title="Avg % change")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each ISO week bucket pools the same calendar week across ~11 years (e.g. all 'week 15' sessions from "
        "2015-2026), so it has far fewer observations per bucket than the month or weekday cuts — read it as "
        "exploratory, not conclusive."
    )

    st.divider()
    st.header("📋 Summary report — Seasonality")
    st.markdown(build_seasonality_report(month_sum, wom_sum, woy_sum))

    st.subheader("Full monthly statistics")
    display_month = month_sum.rename(columns={
        "trading_days": "Trading Days", "mean_return": "Mean Return %", "median_return": "Median Return %",
        "std_dev": "Std Dev %", "best_day": "Best Day %", "worst_day": "Worst Day %",
        "win_rate_pct": "Win Rate %", "cumulative_return_pct": "Cumulative Return %",
    })
    st.dataframe(display_month, width="stretch")

    st.subheader("Full week-in-month statistics")
    display_wom = wom_sum.rename(columns={
        "trading_days": "Trading Days", "mean_return": "Mean Return %", "median_return": "Median Return %",
        "std_dev": "Std Dev %", "best_day": "Best Day %", "worst_day": "Worst Day %",
        "win_rate_pct": "Win Rate %", "cumulative_return_pct": "Cumulative Return %",
    })
    st.dataframe(display_wom, width="stretch")

    with st.expander("Full week-of-year statistics (53 ISO weeks)"):
        display_woy = woy_sum.rename(columns={
            "trading_days": "Trading Days", "mean_return": "Mean Return %", "median_return": "Median Return %",
            "std_dev": "Std Dev %", "best_day": "Best Day %", "worst_day": "Worst Day %",
            "win_rate_pct": "Win Rate %", "cumulative_return_pct": "Cumulative Return %",
        })
        st.dataframe(display_woy, width="stretch")

with st.expander("Methodology"):
    st.markdown(
        """
- **Universe:** Nifty 50 index (`^NSEI`), daily OHLC from Yahoo Finance, 1 Jan 2015 – 16 Jun 2026.
- **Return definition:** `% change = (Close_t - Close_t-1) / Close_t-1 * 100`, using the previous *trading* day's
  close (so the first session after a holiday compares against the last session before it — no gap-filling).
- **Weekday assignment:** taken directly from each row's calendar date (`date.day_name()`), not assumed.
- **Month assignment:** calendar month name (`date.month_name()`), aggregated across all years in range.
- **Week-of-month (turn-of-month):** `(day_of_month - 1) // 7 + 1` → buckets 1-5 per month. Week 5 has fewer
  observations since only longer months reach it.
- **Week-of-year:** ISO calendar week number (1-53), pooling the same week across every year in range.
- **Win rate:** % of sessions in that bucket that closed higher than the previous close.
- **Cumulative return:** compounding only the returns observed in that specific bucket across the whole period
  (a hypothetical "I only get exposure in April" scenario) — not a realistic strategy, just a way to see which
  bucket's tilt accumulated the most over time.
        """
    )
