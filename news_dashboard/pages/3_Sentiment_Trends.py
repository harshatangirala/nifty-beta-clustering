"""Page 3 – Sentiment Trends & Time-Series Analysis"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from config import DB_PATH, CONTINENTS, SENTIMENT_COLORS

st.set_page_config(page_title="Sentiment Trends", page_icon="📈", layout="wide")

# ── Resources ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Sentiment Trends")
today     = date.today()
from_date = st.sidebar.date_input("From", today - timedelta(days=30), max_value=today)
to_date   = st.sidebar.date_input("To",   today,                      max_value=today)

continent_filter = st.sidebar.multiselect(
    "Continents", list(CONTINENTS.keys()),
    default=list(CONTINENTS.keys()),
)
ma_window = st.sidebar.slider("Moving-Average window (days)", 1, 14, 3)
granularity = st.sidebar.radio("Granularity", ["Daily", "Weekly"], index=0)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_ts_data(from_d, to_d):
    db = get_db()
    return db.get_sentiment_timeseries(from_date=str(from_d), to_date=str(to_d))

@st.cache_data(ttl=300)
def load_articles(from_d, to_d):
    db = get_db()
    return db.get_articles(from_date=str(from_d), to_date=str(to_d), limit=2000)

ts_raw  = load_ts_data(from_date, to_date)
art_df  = load_articles(from_date, to_date)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2>📈 Sentiment Trends & Time-Series Analysis</h2>"
    f"<p style='color:#9AA0B4'>{str(from_date)} → {str(to_date)}</p>",
    unsafe_allow_html=True,
)

if ts_raw.empty and art_df.empty:
    st.info("No data available. Fetch news from Home or Continental News first.")
    st.stop()

# ── Pre-process time-series data ───────────────────────────────────────────────

def resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate daily data to weekly if needed."""
    if df.empty:
        return df
    df = df.copy()
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    if freq == "Weekly":
        df["pub_date"] = df["pub_date"].dt.to_period("W").dt.start_time
    return df.groupby(["pub_date", "continent", "sentiment_label"], as_index=False).agg(
        article_count=("article_count", "sum"),
        avg_positive =("avg_positive",  "mean"),
        avg_negative =("avg_negative",  "mean"),
        avg_neutral  =("avg_neutral",   "mean"),
        avg_compound =("avg_compound",  "mean"),
    )

ts = resample_df(ts_raw, granularity)
if continent_filter:
    ts = ts[ts["continent"].isin(continent_filter)]

# ── Overall daily compound score ───────────────────────────────────────────────

st.subheader("🌐 Overall Market Sentiment Score")

if not ts.empty and "avg_compound" in ts.columns:
    overall = (
        ts.groupby("pub_date", as_index=False)
          .agg(avg_compound=("avg_compound", "mean"),
               article_count=("article_count", "sum"))
    )
    overall = overall.sort_values("pub_date")
    overall[f"MA{ma_window}"] = overall["avg_compound"].rolling(ma_window, min_periods=1).mean()

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Bar(
        x=overall["pub_date"], y=overall["avg_compound"],
        name="Daily Compound",
        marker_color=[
            "#2ECC71" if v >= 0.05 else ("#E74C3C" if v <= -0.05 else "#95A5A6")
            for v in overall["avg_compound"]
        ],
        opacity=0.6,
    ))
    fig_ov.add_trace(go.Scatter(
        x=overall["pub_date"], y=overall[f"MA{ma_window}"],
        name=f"{ma_window}-Day MA",
        line=dict(color="#F39C12", width=2),
    ))
    fig_ov.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1)
    fig_ov.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=300,
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(gridcolor="#2D3153"),
        yaxis=dict(gridcolor="#2D3153", title="Compound Score [-1→+1]"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ov, use_container_width=True)

# ── Continental comparison ─────────────────────────────────────────────────────

st.subheader("🌍 Continental Sentiment Comparison")

if not ts.empty:
    cont_daily = (
        ts.groupby(["pub_date", "continent"], as_index=False)
          .agg(avg_compound=("avg_compound", "mean"),
               article_count=("article_count", "sum"))
    )
    fig_cont = px.line(
        cont_daily, x="pub_date", y="avg_compound",
        color="continent",
        color_discrete_map={
            "Asia":          "#FF6B6B",
            "Europe":        "#4ECDC4",
            "North America": "#45B7D1",
        },
        markers=True,
        labels={"avg_compound": "Avg Compound Score", "pub_date": "Date",
                "continent": "Continent"},
    )
    fig_cont.add_hline(y=0, line_dash="dot", line_color="#9AA0B4", line_width=1)
    fig_cont.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=320,
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(gridcolor="#2D3153"),
        yaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cont, use_container_width=True)

# ── Stacked area: article volume by sentiment ──────────────────────────────────

st.subheader("📊 News Volume by Sentiment")

if not ts.empty:
    vol_df = (
        ts.groupby(["pub_date", "sentiment_label"], as_index=False)
          .agg(count=("article_count", "sum"))
    )
    fig_area = px.area(
        vol_df, x="pub_date", y="count",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"count": "Articles", "pub_date": "Date", "sentiment_label": "Sentiment"},
    )
    fig_area.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=300,
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(gridcolor="#2D3153"),
        yaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_area, use_container_width=True)

# ── Positive/Negative ratio ────────────────────────────────────────────────────

st.subheader("⚖️ Bullish-to-Bearish Ratio")

if not ts.empty:
    ratio_df = (
        ts.groupby(["pub_date", "sentiment_label"], as_index=False)
          .agg(count=("article_count", "sum"))
    )
    piv = ratio_df.pivot(index="pub_date", columns="sentiment_label", values="count").fillna(0)
    if "positive" in piv.columns and "negative" in piv.columns:
        piv["ratio"] = piv["positive"] / (piv["negative"] + 0.001)
        piv = piv.reset_index()
        piv[f"MA{ma_window}"] = piv["ratio"].rolling(ma_window, min_periods=1).mean()

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Bar(
            x=piv["pub_date"], y=piv["ratio"],
            name="Bull/Bear Ratio",
            marker_color=[
                "#2ECC71" if v >= 1 else "#E74C3C"
                for v in piv["ratio"]
            ],
            opacity=0.6,
        ))
        fig_ratio.add_trace(go.Scatter(
            x=piv["pub_date"], y=piv[f"MA{ma_window}"],
            name=f"{ma_window}-Day MA",
            line=dict(color="#F39C12", width=2),
        ))
        fig_ratio.add_hline(y=1, line_dash="dash", line_color="#9AA0B4", line_width=1,
                            annotation_text="Neutral (1.0)")
        fig_ratio.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=280,
            legend=dict(orientation="h", y=-0.2),
            xaxis=dict(gridcolor="#2D3153"),
            yaxis=dict(gridcolor="#2D3153", title="Positive / Negative"),
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

# ── Country-level heatmap over time ───────────────────────────────────────────

st.subheader("🗓️ Country Sentiment Calendar Heatmap")

if not art_df.empty and "country" in art_df.columns:
    art_df["date"] = pd.to_datetime(art_df["published_at"], errors="coerce").dt.date
    cal_df = (
        art_df.dropna(subset=["date", "compound_score", "country"])
              .groupby(["country", "date"], as_index=False)
              .agg(avg_compound=("compound_score", "mean"),
                   n=("title", "count"))
    )
    # Pivot: countries as rows, dates as columns
    top_countries = cal_df.groupby("country")["n"].sum().nlargest(15).index
    cal_df = cal_df[cal_df["country"].isin(top_countries)]
    pivot_cal = cal_df.pivot(index="country", columns="date", values="avg_compound")

    fig_cal = go.Figure(go.Heatmap(
        z=pivot_cal.values,
        x=[str(c) for c in pivot_cal.columns],
        y=pivot_cal.index.tolist(),
        colorscale=[[0, "#E74C3C"], [0.5, "#2B2E40"], [1, "#2ECC71"]],
        zmid=0,
        colorbar=dict(title="Compound", tickfont=dict(color="#E8EAED")),
        hovertemplate="Country: %{y}<br>Date: %{x}<br>Score: %{z:.3f}<extra></extra>",
    ))
    fig_cal.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=max(300, len(top_countries) * 22 + 80),
        xaxis=dict(tickangle=-45, nticks=14),
        margin=dict(l=0, r=10, t=10, b=60),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ── Industry sentiment trends ──────────────────────────────────────────────────

st.subheader("🏭 Industry Sentiment Trends")

@st.cache_data(ttl=300)
def load_stock_data_trends(from_d, to_d):
    db = get_db()
    return db.get_stock_impact(from_date=str(from_d), to_date=str(to_d))

si_df = load_stock_data_trends(from_date, to_date)

if not si_df.empty and "industry" in si_df.columns:
    si_df["date"] = pd.to_datetime(si_df["published_at"], errors="coerce").dt.date
    ind_ts = (
        si_df.dropna(subset=["date", "industry", "compound_score"])
             .groupby(["industry", "date"], as_index=False)
             .agg(avg_compound=("compound_score", "mean"),
                  n=("symbol", "count"))
    )
    top_ind = ind_ts.groupby("industry")["n"].sum().nlargest(8).index
    ind_ts  = ind_ts[ind_ts["industry"].isin(top_ind)]

    fig_ind = px.line(
        ind_ts, x="date", y="avg_compound", color="industry",
        markers=False,
        labels={"avg_compound": "Avg Compound Score", "date": "Date", "industry": "Industry"},
    )
    fig_ind.add_hline(y=0, line_dash="dot", line_color="#9AA0B4", line_width=1)
    fig_ind.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=320,
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        xaxis=dict(gridcolor="#2D3153"),
        yaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ind, use_container_width=True)
else:
    st.info("No stock-industry data available.")

# ── Compound score statistics table ───────────────────────────────────────────

with st.expander("📋 Summary Statistics Table"):
    if not ts.empty:
        stat_df = (
            ts.groupby(["continent", "sentiment_label"])
              .agg(
                  total_articles=("article_count", "sum"),
                  avg_compound   =("avg_compound", "mean"),
                  min_compound   =("avg_compound", "min"),
                  max_compound   =("avg_compound", "max"),
                  std_compound   =("avg_compound", "std"),
              )
              .reset_index()
        )
        st.dataframe(
            stat_df.style.format({
                "avg_compound": "{:+.3f}",
                "min_compound": "{:+.3f}",
                "max_compound": "{:+.3f}",
                "std_compound": "{:.3f}",
            }),
            use_container_width=True,
        )
