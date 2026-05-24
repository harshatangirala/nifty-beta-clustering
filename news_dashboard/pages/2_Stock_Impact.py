"""
Page 2 – Stock Impact Analysis
NSE (India) + S&P 500 (USA): news-driven sentiment metrics for every stock.

Metrics per stock
-----------------
  Sentiment Score   : avg FinBERT/VADER compound score  [-1 → +1]
  News Count        : total articles mentioning the stock
  Positive %        : share of positive articles
  Negative %        : share of negative articles
  Impact Rating     : Bullish / Bearish / Neutral label
  Weighted Impact   : compound × relevance_score (emphasis on close matches)
  Momentum          : difference between recent 3-day avg and overall avg
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from config import DB_PATH, NSE_STOCKS_CSV, SENTIMENT_COLORS, SENTIMENT_EMOJIS

st.set_page_config(page_title="Stock Impact Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
.news-card{background:#1A1D2E;border-radius:10px;padding:14px 18px;
           border:1px solid #2D3153;margin-bottom:10px;}
.badge-positive{background:#1B4B2E;color:#2ECC71;border-radius:6px;
                padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-negative{background:#4B1B1B;color:#E74C3C;border-radius:6px;
                padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-neutral{background:#2B2E40;color:#95A5A6;border-radius:6px;
               padding:3px 10px;font-size:.78rem;font-weight:700;}
.rating-bullish{color:#2ECC71;font-weight:700;}
.rating-bearish{color:#E74C3C;font-weight:700;}
.rating-neutral{color:#95A5A6;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

@st.cache_resource
def get_nse_universe():
    p = NSE_STOCKS_CSV
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Symbol","Company Name","Industry","Index Name"])

@st.cache_resource
def get_sp500_universe():
    from src.sp500_stocks import get_sp500_df as _f
    return _f()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Stock Impact")
today     = date.today()
from_date = st.sidebar.date_input("From", today - timedelta(days=14), max_value=today)
to_date   = st.sidebar.date_input("To",   today,                      max_value=today)

market_tab = st.sidebar.radio("Market", ["NSE – India 🇮🇳", "S&P 500 – USA 🇺🇸", "Both 🌐"], index=0)
market_key = (
    "NSE"     if "NSE" in market_tab else
    "S&P 500" if "S&P" in market_tab else
    None
)

nse_df  = get_nse_universe()
sp5_df  = get_sp500_universe()

all_sectors = sorted(set(
    list(nse_df["Industry"].dropna().unique()) +
    list(sp5_df["Industry"].dropna().unique())
))
sector_filter = st.sidebar.multiselect("Filter by Sector", all_sectors, default=[])

top_n         = st.sidebar.slider("Top N stocks in charts", 10, 50, 20, 5)
min_relevance = st.sidebar.slider("Min relevance score",    1.0, 15.0, 2.0, 0.5)
min_articles  = st.sidebar.slider("Min articles",           1, 10, 1, 1)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_stock_data(mkt, from_d, to_d):
    return get_db().get_stock_impact(
        from_date=str(from_d), to_date=str(to_d), market=mkt
    )

raw = load_stock_data(market_key, from_date, to_date)

# Apply filters
if not raw.empty:
    if sector_filter and "industry" in raw.columns:
        raw = raw[raw["industry"].isin(sector_filter)]
    if "relevance_score" in raw.columns:
        raw = raw[pd.to_numeric(raw["relevance_score"], errors="coerce").fillna(0) >= min_relevance]

# ── Header ─────────────────────────────────────────────────────────────────────

mkt_icon = "🇮🇳" if "NSE" in market_tab else ("🇺🇸" if "S&P" in market_tab else "🌐")
st.markdown(
    f"<h2>{mkt_icon} Stock Impact Analysis — {market_tab}</h2>"
    f"<p style='color:#9AA0B4'>{from_date} → {to_date}</p>",
    unsafe_allow_html=True,
)

if raw.empty:
    st.info(
        "No stock-impact data found for the selected filters. "
        "Go to the **Home** page and click **Fetch & Analyse News** first."
    )
    st.stop()

# ── Build per-stock metrics ───────────────────────────────────────────────────

def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level rows into one row per stock with full metrics."""
    df = df.copy()
    df["compound_score"]  = pd.to_numeric(df["compound_score"],  errors="coerce").fillna(0)
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce").fillna(1)
    df["positive_score"]  = pd.to_numeric(df["positive_score"],  errors="coerce").fillna(0)
    df["negative_score"]  = pd.to_numeric(df["negative_score"],  errors="coerce").fillna(0)
    df["neutral_score"]   = pd.to_numeric(df["neutral_score"],   errors="coerce").fillna(0)
    df["published_at"]    = pd.to_datetime(df["published_at"],   errors="coerce")

    recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    if df["published_at"].dt.tz is None:
        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=3)

    agg = (
        df.groupby(["symbol","company_name","market","industry"], as_index=False)
        .agg(
            news_count      =("title",          "count"),
            avg_compound    =("compound_score",  "mean"),
            avg_positive    =("positive_score",  "mean"),
            avg_negative    =("negative_score",  "mean"),
            avg_neutral     =("neutral_score",   "mean"),
            positive_count  =("sentiment_label", lambda x: (x=="positive").sum()),
            negative_count  =("sentiment_label", lambda x: (x=="negative").sum()),
            neutral_count   =("sentiment_label", lambda x: (x=="neutral").sum()),
            avg_relevance   =("relevance_score", "mean"),
            weighted_impact =("compound_score",  lambda x:
                              np.average(x, weights=df.loc[x.index,"relevance_score"])),
        )
    )

    # Momentum = compound of last 3 days minus overall
    recent = df[df["published_at"] >= recent_cutoff]
    if not recent.empty:
        mom = recent.groupby("symbol")["compound_score"].mean().rename("recent_compound")
        agg = agg.merge(mom, on="symbol", how="left")
        agg["momentum"] = (agg["recent_compound"] - agg["avg_compound"]).fillna(0)
    else:
        agg["momentum"] = 0.0

    agg["positive_pct"] = agg["positive_count"] / agg["news_count"].clip(lower=1) * 100
    agg["negative_pct"] = agg["negative_count"] / agg["news_count"].clip(lower=1) * 100
    agg["neutral_pct"]  = agg["neutral_count"]  / agg["news_count"].clip(lower=1) * 100

    def impact_rating(row):
        c = row["avg_compound"]
        if c >= 0.15:
            return "Bullish"
        if c <= -0.15:
            return "Bearish"
        return "Neutral"

    agg["impact_rating"] = agg.apply(impact_rating, axis=1)

    return agg.sort_values("news_count", ascending=False)

all_metrics = build_metrics(raw)
all_metrics = all_metrics[all_metrics["news_count"] >= min_articles]

# ── Split NSE / S&P500 ────────────────────────────────────────────────────────

nse_met = all_metrics[all_metrics["market"] == "NSE"]      if market_key != "S&P 500" else pd.DataFrame()
sp5_met = all_metrics[all_metrics["market"] == "S&P 500"]  if market_key != "NSE"     else pd.DataFrame()
if market_key is None:
    nse_met = all_metrics[all_metrics["market"] == "NSE"]
    sp5_met = all_metrics[all_metrics["market"] == "S&P 500"]

# ── Summary KPIs ──────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stocks Tracked",   all_metrics["symbol"].nunique())
c2.metric("Total Mentions",   int(all_metrics["news_count"].sum()))
c3.metric("Bullish Stocks 📈", int((all_metrics["impact_rating"]=="Bullish").sum()))
c4.metric("Bearish Stocks 📉", int((all_metrics["impact_rating"]=="Bearish").sum()))
c5.metric("Avg Sentiment",    f"{all_metrics['avg_compound'].mean():+.3f}")
c6.metric("Avg Weighted Impact", f"{all_metrics['weighted_impact'].mean():+.3f}")

st.markdown("---")

# ── Colour helper ─────────────────────────────────────────────────────────────

def _safe_label(val):
    return val if isinstance(val, str) and val in ("positive","negative","neutral") else "neutral"

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET TABS
# ═══════════════════════════════════════════════════════════════════════════════

def render_market(df_met: pd.DataFrame, df_raw: pd.DataFrame,
                  market_name: str, flag: str, universe_df: pd.DataFrame):
    if df_met.empty:
        st.info(f"No {market_name} stocks found in cached news.")
        return

    st.markdown(f"## {flag} {market_name} Stock Impact")

    top = df_met.head(top_n)

    # ── Impact leaderboard ────────────────────────────────────────────────────
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.subheader("📈 Most Bullish")
        bullish = df_met[df_met["impact_rating"]=="Bullish"].nlargest(8,"avg_compound")
        if not bullish.empty:
            fig = px.bar(
                bullish.sort_values("avg_compound"),
                x="avg_compound", y="symbol", orientation="h",
                color="avg_compound",
                color_continuous_scale=["#2B2E40","#2ECC71"],
                range_color=[0,1],
                text=bullish.sort_values("avg_compound")["avg_compound"].map("{:+.3f}".format),
                labels={"avg_compound":"Sentiment Score","symbol":""},
                hover_data={"company_name":True,"news_count":True,"positive_pct":":.1f"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), height=320,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#2D3153", range=[0,1]),
                margin=dict(l=0,r=20,t=10,b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_bear:
        st.subheader("📉 Most Bearish")
        bearish = df_met[df_met["impact_rating"]=="Bearish"].nsmallest(8,"avg_compound")
        if not bearish.empty:
            fig2 = px.bar(
                bearish.sort_values("avg_compound", ascending=False),
                x="avg_compound", y="symbol", orientation="h",
                color="avg_compound",
                color_continuous_scale=["#E74C3C","#2B2E40"],
                range_color=[-1,0],
                text=bearish.sort_values("avg_compound",ascending=False)["avg_compound"].map("{:+.3f}".format),
                labels={"avg_compound":"Sentiment Score","symbol":""},
                hover_data={"company_name":True,"news_count":True,"negative_pct":":.1f"},
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), height=320,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#2D3153", range=[-1,0]),
                margin=dict(l=0,r=20,t=10,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Bubble chart: volume vs sentiment ─────────────────────────────────────

    st.subheader("🫧 News Volume vs Sentiment Score")
    fig_bub = px.scatter(
        top,
        x="avg_compound", y="avg_positive",
        size="news_count",
        color="impact_rating",
        color_discrete_map={"Bullish":"#2ECC71","Bearish":"#E74C3C","Neutral":"#95A5A6"},
        hover_name="symbol",
        hover_data={
            "company_name":True,"industry":True,
            "news_count":True,"positive_pct":":.1f","negative_pct":":.1f",
            "avg_compound":":.3f","weighted_impact":":.3f",
        },
        size_max=40,
        labels={
            "avg_compound":"Avg Compound Score [-1→+1]",
            "avg_positive":"Avg Positive Probability",
        },
    )
    fig_bub.add_vline(x=0,  line_dash="dash", line_color="#9AA0B4", line_width=1)
    fig_bub.add_hline(y=0.5, line_dash="dash", line_color="#9AA0B4", line_width=1)
    fig_bub.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=400,
        legend=dict(orientation="h",y=-0.18),
        xaxis=dict(gridcolor="#2D3153"),
        yaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_bub, use_container_width=True)

    # ── Stacked sentiment heatmap ─────────────────────────────────────────────

    st.subheader("🗺️ Positive / Negative / Neutral Heatmap")
    hm_df = top.set_index("symbol")[["avg_positive","avg_negative","avg_neutral"]].T
    hm_df.index = ["Positive","Negative","Neutral"]
    fig_hm = go.Figure(go.Heatmap(
        z=hm_df.values,
        x=hm_df.columns.tolist(),
        y=hm_df.index.tolist(),
        colorscale=[[0,"#1A1D2E"],[0.5,"#2B2E40"],[1,"#2ECC71"]],
        text=[[f"{v:.2f}" for v in row] for row in hm_df.values],
        texttemplate="%{text}",
        colorbar=dict(title="Prob", tickfont=dict(color="#E8EAED")),
        hovertemplate="Stock: %{x}<br>%{y}: %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=220,
        xaxis=dict(tickangle=-45),
        margin=dict(l=10,r=10,t=10,b=60),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Momentum chart ────────────────────────────────────────────────────────

    st.subheader("⚡ Sentiment Momentum (Recent 3-day vs Overall)")
    mom_df = top.sort_values("momentum", ascending=False)
    fig_mom = px.bar(
        mom_df,
        x="symbol", y="momentum",
        color="momentum",
        color_continuous_scale=["#E74C3C","#2B2E40","#2ECC71"],
        color_continuous_midpoint=0,
        hover_data={"company_name":True,"avg_compound":":.3f","news_count":True},
        labels={"momentum":"Momentum (3d avg − overall avg)","symbol":""},
    )
    fig_mom.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1)
    fig_mom.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=300,
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_mom, use_container_width=True)

    # ── Sector breakdown ──────────────────────────────────────────────────────

    st.subheader("🏭 Sector Sentiment Breakdown")
    sec_df = (
        df_met.groupby("industry", as_index=False)
        .agg(
            stocks      =("symbol",       "nunique"),
            news_count  =("news_count",   "sum"),
            avg_compound=("avg_compound", "mean"),
            bullish     =("impact_rating", lambda x:(x=="Bullish").sum()),
            bearish     =("impact_rating", lambda x:(x=="Bearish").sum()),
        )
        .sort_values("avg_compound", ascending=False)
        .head(15)
    )
    fig_sec = px.bar(
        sec_df, x="avg_compound", y="industry", orientation="h",
        color="avg_compound",
        color_continuous_scale=["#E74C3C","#2B2E40","#2ECC71"],
        color_continuous_midpoint=0,
        text=sec_df["avg_compound"].map("{:+.3f}".format),
        hover_data={"stocks":True,"news_count":True,"bullish":True,"bearish":True},
        labels={"avg_compound":"Avg Compound Score","industry":""},
    )
    fig_sec.update_traces(textposition="outside")
    fig_sec.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=480,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0,r=40,t=10,b=0),
    )
    st.plotly_chart(fig_sec, use_container_width=True)

    # ── Full metrics table ────────────────────────────────────────────────────

    st.subheader("📋 Full Stock Metrics Table")

    def colour_rating(val):
        if val == "Bullish": return "color: #2ECC71; font-weight: bold"
        if val == "Bearish": return "color: #E74C3C; font-weight: bold"
        return "color: #95A5A6"

    def colour_compound(val):
        try:
            v = float(val)
            if v >= 0.1:  return "color: #2ECC71"
            if v <= -0.1: return "color: #E74C3C"
        except: pass
        return "color: #95A5A6"

    display = df_met[[
        "symbol","company_name","industry","news_count",
        "avg_compound","weighted_impact","momentum",
        "positive_pct","negative_pct","neutral_pct",
        "impact_rating",
    ]].rename(columns={
        "symbol":         "Ticker",
        "company_name":   "Company",
        "industry":       "Sector",
        "news_count":     "Articles",
        "avg_compound":   "Sentiment Score",
        "weighted_impact":"Weighted Impact",
        "momentum":       "Momentum",
        "positive_pct":   "Positive %",
        "negative_pct":   "Negative %",
        "neutral_pct":    "Neutral %",
        "impact_rating":  "Rating",
    })

    # Search box
    search = st.text_input(f"Search {market_name} stocks", "", key=f"search_{market_name}")
    if search:
        mask = (
            display["Ticker"].str.contains(search, case=False, na=False) |
            display["Company"].str.contains(search, case=False, na=False) |
            display["Sector"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    styled = (
        display.style
        .format({
            "Sentiment Score": "{:+.3f}",
            "Weighted Impact": "{:+.3f}",
            "Momentum":        "{:+.3f}",
            "Positive %":      "{:.1f}%",
            "Negative %":      "{:.1f}%",
            "Neutral %":       "{:.1f}%",
        })
        .map(colour_rating,   subset=["Rating"])
        .map(colour_compound, subset=["Sentiment Score","Weighted Impact","Momentum"])
    )
    st.dataframe(styled, use_container_width=True, height=420)

    with st.expander("ℹ️ Metric definitions"):
        st.markdown("""
| Metric | Meaning |
|--------|---------|
| **Sentiment Score** | Avg FinBERT/VADER compound score: +1 = fully positive, −1 = fully negative |
| **Weighted Impact** | Compound score weighted by match relevance (exact ticker hit scores higher) |
| **Momentum** | Recent 3-day avg compound minus overall avg — positive = improving sentiment |
| **Positive / Negative %** | Share of articles with that label |
| **Rating** | Bullish (score ≥ +0.15) · Bearish (≤ −0.15) · Neutral otherwise |
""")

    # ── Individual stock drill-down ───────────────────────────────────────────

    st.markdown("---")
    st.subheader("🔍 Stock News Feed")

    stock_options = ["— All stocks —"] + sorted(df_met["symbol"].tolist())
    sel = st.selectbox(f"Select {market_name} stock", stock_options, key=f"sel_{market_name}")

    sub = df_raw.copy()
    if "compound_score" in sub.columns:
        sub["compound_score"] = pd.to_numeric(sub["compound_score"], errors="coerce").fillna(0)
    if sel != "— All stocks —":
        sub = sub[sub["symbol"] == sel]

    sub = sub.sort_values("published_at", ascending=False)

    if not sub.empty and sel != "— All stocks —":
        m1, m2, m3, m4 = st.columns(4)
        pos_p = (sub["sentiment_label"] == "positive").mean() * 100
        neg_p = (sub["sentiment_label"] == "negative").mean() * 100
        avg_c = sub["compound_score"].mean()
        avg_r = pd.to_numeric(sub["relevance_score"], errors="coerce").mean()
        m1.metric("Articles",       len(sub))
        m2.metric("Positive %",     f"{pos_p:.1f}%")
        m3.metric("Negative %",     f"{neg_p:.1f}%")
        m4.metric("Avg Sentiment",  f"{avg_c:+.3f}")

        # Mini timeline
        sub["date"] = pd.to_datetime(sub["published_at"], errors="coerce").dt.date
        tl = (
            sub.dropna(subset=["date","compound_score"])
               .groupby("date")["compound_score"]
               .mean().reset_index()
               .sort_values("date")
        )
        if len(tl) > 1:
            fig_tl = px.line(
                tl, x="date", y="compound_score",
                labels={"compound_score":"Sentiment","date":"Date"},
                markers=True, title=f"{sel} — Sentiment Over Time",
                color_discrete_sequence=["#4ECDC4"],
            )
            fig_tl.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1)
            fig_tl.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), height=240,
                xaxis=dict(gridcolor="#2D3153"),
                yaxis=dict(gridcolor="#2D3153"),
                margin=dict(l=0,r=0,t=30,b=0),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(sub.head(30).iterrows()):
        _sl = row.get("sentiment_label")
        sentiment = _sl if isinstance(_sl, str) and _sl in ("positive","negative","neutral") else "neutral"
        badge_cls = f"badge-{sentiment}"
        emoji_s   = SENTIMENT_EMOJIS.get(sentiment, "➡️")
        title     = (row.get("title") or "")[:130]
        url       = row.get("url") or "#"
        pub       = str(row.get("published_at") or "")[:16].replace("T"," ")
        compound  = float(row.get("compound_score") or 0)
        rel_sc    = float(row.get("relevance_score") or 0)
        country   = row.get("country") or ""
        src       = row.get("source_name") or ""

        with (col1 if i%2==0 else col2):
            st.markdown(f"""
<div class='news-card'>
  <span class='{badge_cls}'>{emoji_s} {sentiment.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{country} · {pub} · {src}</small>
  <small style='color:#F39C12;float:right'>⭐ {rel_sc:.1f}</small><br><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:700;font-size:.9rem;'>
    {title}
  </a><br>
  <small style='color:#F39C12'>Compound: {compound:+.3f}</small>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER TABS
# ═══════════════════════════════════════════════════════════════════════════════

if market_key == "NSE" or market_key is None:
    render_market(nse_met, raw[raw["market"]=="NSE"] if "market" in raw.columns else raw,
                  "NSE – India", "🇮🇳", nse_df)

if market_key == "S&P 500" or market_key is None:
    if market_key is None:
        st.markdown("---")
    render_market(sp5_met, raw[raw["market"]=="S&P 500"] if "market" in raw.columns else raw,
                  "S&P 500 – USA", "🇺🇸", sp5_df)
