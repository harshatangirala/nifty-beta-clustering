"""Page 2 – Stock Impact Analysis (NSE + S&P 500)"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from config import DB_PATH, NSE_STOCKS_CSV, SENTIMENT_COLORS, SENTIMENT_EMOJIS

st.set_page_config(page_title="Stock Impact Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stock-card { background:#1A1D2E; border-radius:10px; padding:14px 18px;
              border:1px solid #2D3153; margin-bottom:10px; }
.badge-positive { background:#1B4B2E; color:#2ECC71; border-radius:6px; padding:3px 10px;
                  font-size:.78rem; font-weight:700; }
.badge-negative { background:#4B1B1B; color:#E74C3C; border-radius:6px; padding:3px 10px;
                  font-size:.78rem; font-weight:700; }
.badge-neutral  { background:#2B2E40; color:#95A5A6; border-radius:6px; padding:3px 10px;
                  font-size:.78rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

@st.cache_resource
def get_nse_df():
    p = NSE_STOCKS_CSV
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Symbol", "Company Name", "Industry", "Index Name"])

@st.cache_resource
def get_sp500_df():
    from src.sp500_stocks import get_sp500_df as _f
    return _f()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Stock Impact")
market = st.sidebar.radio("Market", ["NSE (India)", "S&P 500 (USA)", "Both"], index=0)

today      = date.today()
from_date  = st.sidebar.date_input("From", today - timedelta(days=7), max_value=today)
to_date    = st.sidebar.date_input("To",   today,                     max_value=today)

# Sector filter
nse_df   = get_nse_df()
sp5_df   = get_sp500_df()
all_sectors = sorted(set(
    list(nse_df["Industry"].dropna().unique()) +
    list(sp5_df["Industry"].dropna().unique())
))
sector_filter = st.sidebar.multiselect("Sector / Industry", all_sectors, default=[])

top_n = st.sidebar.slider("Top N stocks", 5, 50, 20, 5)

st.sidebar.markdown("---")
min_relevance = st.sidebar.slider("Min Relevance Score", 1.0, 20.0, 2.0, 0.5)
st.sidebar.caption("Higher = stricter stock matching")

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_stock_data(mkt, from_d, to_d):
    db  = get_db()
    mkt_filter = None
    if mkt == "NSE (India)":
        mkt_filter = "NSE"
    elif mkt == "S&P 500 (USA)":
        mkt_filter = "S&P 500"
    return db.get_stock_impact(
        from_date=str(from_d), to_date=str(to_d), market=mkt_filter
    )

df = load_stock_data(market, from_date, to_date)

# Apply sector filter
if sector_filter and "industry" in df.columns:
    df = df[df["industry"].isin(sector_filter)]

# Apply relevance filter
if "relevance_score" in df.columns:
    df = df[df["relevance_score"] >= min_relevance]

# ── Header ─────────────────────────────────────────────────────────────────────

mkt_icon = "🇮🇳" if "NSE" in market else ("🇺🇸" if "S&P" in market else "🌐")
st.markdown(
    f"<h2>{mkt_icon} Stock Impact Analysis — {market}</h2>"
    f"<p style='color:#9AA0B4'>{str(from_date)} → {str(to_date)} · "
    f"{df['symbol'].nunique() if 'symbol' in df.columns else 0} stocks impacted</p>",
    unsafe_allow_html=True,
)

if df.empty:
    st.info(
        "No stock-impact data found. "
        "Fetch news first from the Home page or Continental News page."
    )
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────

total_mentions = len(df)
unique_stocks  = df["symbol"].nunique() if "symbol" in df.columns else 0
most_positive  = df[df["sentiment_label"] == "positive"]["symbol"].value_counts().idxmax() \
                 if (df["sentiment_label"] == "positive").any() else "—"
most_negative  = df[df["sentiment_label"] == "negative"]["symbol"].value_counts().idxmax() \
                 if (df["sentiment_label"] == "negative").any() else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Mentions",  total_mentions)
c2.metric("Unique Stocks",   unique_stocks)
c3.metric("Most Bullish 📈", most_positive)
c4.metric("Most Bearish 📉", most_negative)
st.markdown("---")

# ── Stock aggregation ─────────────────────────────────────────────────────────

agg = (
    df.groupby(["symbol", "company_name", "market", "industry"], as_index=False)
    .agg(
        article_count   =("title",        "count"),
        avg_relevance   =("relevance_score","mean"),
        avg_compound    =("compound_score", "mean"),
        avg_positive    =("positive_score", "mean"),
        avg_negative    =("negative_score", "mean"),
        positive_count  =("sentiment_label", lambda x: (x == "positive").sum()),
        negative_count  =("sentiment_label", lambda x: (x == "negative").sum()),
        neutral_count   =("sentiment_label", lambda x: (x == "neutral" ).sum()),
    )
    .sort_values("article_count", ascending=False)
)

agg["dominant_sentiment"] = agg.apply(
    lambda r: "positive" if r["positive_count"] >= r["negative_count"]
              else ("negative" if r["negative_count"] > r["neutral_count"] else "neutral"),
    axis=1,
)

top_agg = agg.head(top_n)

# ── Chart row 1 ───────────────────────────────────────────────────────────────

col_bar, col_sentiment = st.columns(2)

with col_bar:
    st.subheader(f"🏆 Top {top_n} Most-Mentioned Stocks")
    fig_top = px.bar(
        top_agg.sort_values("article_count"),
        x="article_count", y="symbol",
        orientation="h",
        color="dominant_sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hover_data={"company_name": True, "industry": True, "avg_compound": ":.3f"},
        labels={"article_count": "Articles", "symbol": ""},
    )
    fig_top.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=400,
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(gridcolor="#2D3153"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_top, use_container_width=True)

with col_sentiment:
    st.subheader("📈 Sentiment Scores per Stock")
    fig_scatter = px.scatter(
        top_agg, x="avg_positive", y="avg_negative",
        size="article_count",
        color="dominant_sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hover_name="symbol",
        hover_data={"company_name": True, "industry": True, "article_count": True, "avg_compound": ":.3f"},
        labels={"avg_positive": "Avg Positive Prob", "avg_negative": "Avg Negative Prob"},
        size_max=30,
    )
    fig_scatter.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                          line=dict(color="#F39C12", dash="dash", width=1))
    fig_scatter.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font=dict(color="#E8EAED"), height=400,
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(gridcolor="#2D3153", range=[0, 1]),
        yaxis=dict(gridcolor="#2D3153", range=[0, 1]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Sentiment heatmap ─────────────────────────────────────────────────────────

st.subheader("🗺️ Stock Sentiment Heatmap")
pivot = top_agg.set_index("symbol")[["avg_positive", "avg_negative", "avg_neutral", "avg_compound"]]
pivot.columns = ["Positive", "Negative", "Neutral", "Compound"]
fig_heat = go.Figure(go.Heatmap(
    z=pivot.values.T,
    x=pivot.index.tolist(),
    y=["Positive", "Negative", "Neutral", "Compound"],
    colorscale=[
        [0.0, "#4B1B1B"], [0.35, "#2B2E40"],
        [0.5, "#2B2E40"], [0.65, "#2B2E40"],
        [1.0, "#1B4B2E"],
    ],
    zmid=0.33,
    hoverongaps=False,
    text=[[f"{v:.2f}" for v in row] for row in pivot.values.T],
    texttemplate="%{text}",
    colorbar=dict(title="Score", tickfont=dict(color="#E8EAED")),
))
fig_heat.update_layout(
    paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
    font=dict(color="#E8EAED"), height=250,
    xaxis=dict(tickangle=-45),
    margin=dict(l=10, r=10, t=10, b=60),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Sector breakdown ──────────────────────────────────────────────────────────

st.subheader("🏭 Industry Sentiment Breakdown")
sect_df = (
    df.dropna(subset=["industry", "sentiment_label"])
      .groupby(["industry", "sentiment_label"])
      .size().reset_index(name="count")
)
top_sectors = sect_df.groupby("industry")["count"].sum().nlargest(15).index
sect_df = sect_df[sect_df["industry"].isin(top_sectors)]

fig_sect = px.bar(
    sect_df, x="count", y="industry",
    orientation="h",
    color="sentiment_label",
    color_discrete_map=SENTIMENT_COLORS,
    barmode="stack",
    labels={"count": "Articles", "industry": "", "sentiment_label": "Sentiment"},
)
fig_sect.update_layout(
    paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
    font=dict(color="#E8EAED"), height=420,
    legend=dict(orientation="h", y=-0.12),
    xaxis=dict(gridcolor="#2D3153"),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_sect, use_container_width=True)

# ── Individual stock drill-down ───────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Individual Stock News Feed")

if "symbol" in df.columns:
    stock_sel = st.selectbox(
        "Select Stock",
        ["— All —"] + sorted(df["symbol"].unique().tolist()),
    )

    sub = df.copy()
    if stock_sel != "— All —":
        sub = sub[sub["symbol"] == stock_sel]

    sub = sub.sort_values("published_at", ascending=False)

    if not sub.empty:
        summary_cols = st.columns(3)
        pos_pct = (sub["sentiment_label"] == "positive").mean() * 100
        neg_pct = (sub["sentiment_label"] == "negative").mean() * 100
        avg_c   = sub["compound_score"].mean() if "compound_score" in sub else 0
        summary_cols[0].metric("Positive %",    f"{pos_pct:.1f}%")
        summary_cols[1].metric("Negative %",    f"{neg_pct:.1f}%")
        summary_cols[2].metric("Avg Compound",  f"{avg_c:+.3f}")

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(sub.head(40).iterrows()):
        sentiment = row.get("sentiment_label", "neutral") or "neutral"
        badge_cls = f"badge-{sentiment}"
        emoji     = SENTIMENT_EMOJIS.get(sentiment, "➡️")
        title     = row.get("title", "")[:130]
        url       = row.get("url", "#")
        pub       = str(row.get("published_at", ""))[:16].replace("T", " ")
        compound  = float(row.get("compound_score", 0) or 0)
        rel_sc    = float(row.get("relevance_score", 0) or 0)
        country   = row.get("country", "")

        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
<div class='stock-card'>
  <span class='{badge_cls}'>{emoji} {sentiment.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{country} · {pub}</small><br>
  <small style='color:#F39C12'>⭐ Relevance {rel_sc:.1f}</small><br><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:700;font-size:.9rem;'>
    {title}
  </a><br>
  <small style='color:#F39C12'>Compound: {compound:+.3f}</small>
</div>
""", unsafe_allow_html=True)
