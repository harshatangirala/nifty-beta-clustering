"""
Global Financial News Sentiment Dashboard — Home / Overview
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

from config import (
    NEWS_API_KEY, CONTINENTS, SENTIMENT_COLORS, SENTIMENT_EMOJIS,
    PAGE_TITLE, PAGE_ICON, DB_PATH, NSE_STOCKS_CSV,
    CACHE_EXPIRY_HOURS, MAX_ARTICLES_PER_QUERY, DEFAULT_LOOKBACK_DAYS,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1A1D2E; border-radius: 12px; padding: 18px 22px;
    border: 1px solid #2D3153; text-align: center;
}
.metric-card h2 { font-size: 2.2rem; margin: 4px 0; }
.metric-card p  { font-size: 0.82rem; color: #9AA0B4; margin: 0; }
.news-card {
    background: #1A1D2E; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #2D3153; margin-bottom: 10px;
}
.badge-positive { background:#1B4B2E; color:#2ECC71; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.badge-negative { background:#4B1B1B; color:#E74C3C; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.badge-neutral  { background:#2B2E40; color:#95A5A6; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:700; }
.section-header { font-size:1.3rem; font-weight:700; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading database…")
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

@st.cache_resource(show_spinner="Loading stock universe…")
def get_mapper():
    from src.stock_mapper import StockMapper
    from src.sp500_stocks import get_sp500_df

    nse = pd.read_csv(NSE_STOCKS_CSV) if os.path.exists(NSE_STOCKS_CSV) else pd.DataFrame()
    sp5 = get_sp500_df()
    return StockMapper(nse, sp5)

@st.cache_resource(show_spinner="Initialising sentiment engine…")
def get_analyzer():
    from src.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer()

def get_fetcher():
    from src.news_fetcher import NewsAPIFetcher
    return NewsAPIFetcher(NEWS_API_KEY, get_db(), CACHE_EXPIRY_HOURS)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 FinNews Sentiment")
    st.markdown("---")
    st.subheader("🗓️ Global Date Range")
    today       = date.today()
    default_from = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    g_from = st.date_input("From", value=default_from, max_value=today, key="g_from")
    g_to   = st.date_input("To",   value=today,         max_value=today, key="g_to")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    use_finbert = st.checkbox("Use FinBERT (GPU-optional)", value=True,
                              help="Uncheck to use fast VADER baseline only")
    n_articles  = st.slider("Articles per query", 5, 50, MAX_ARTICLES_PER_QUERY, 5)
    auto_refresh = st.checkbox("Auto-refresh (15 min)", value=False)

    st.markdown("---")
    if st.button("🔄 Fetch All Continent News", type="primary", use_container_width=True):
        st.session_state["fetch_all"] = True

    st.markdown("---")
    db_stats = get_db().stats()
    st.metric("Cached Articles",    db_stats["total_articles"])
    st.metric("Sentiment-Analysed", db_stats["analyzed"])
    st.metric("Stock-Mapped",       db_stats["stock_mapped"])
    st.caption("Data powered by [NewsAPI.org](https://newsapi.org)")

# ── Fetch & analyse ────────────────────────────────────────────────────────────

def fetch_and_analyse_all():
    fetcher  = get_fetcher()
    analyzer = get_analyzer()
    mapper   = get_mapper()
    db       = get_db()

    progress = st.progress(0, text="Fetching news…")
    total_q  = sum(len(cfg["global_queries"]) for cfg in CONTINENTS.values())
    done     = 0

    for cont_name, cont_cfg in CONTINENTS.items():
        articles = fetcher.fetch_for_continent(
            cont_name, cont_cfg,
            from_date=g_from, to_date=g_to,
            max_per_query=n_articles,
        )
        texts = [f"{a['title']} {a['description']}" for a in articles]
        model_key = "finbert" if (use_finbert and analyzer.finbert_available()) else "vader"

        if model_key == "finbert":
            sentiments = analyzer.analyze_batch_finbert(texts)
        else:
            sentiments = analyzer.analyze_batch_vader(texts)

        for art, sent in zip(articles, sentiments):
            if not sent:
                continue
            art_id = db.upsert_article(art)
            if art_id:
                db.upsert_sentiment(art_id, model_key, sent)
                related = mapper.find_related_stocks(art["title"], art["description"])
                if related:
                    db.upsert_stock_mentions(art_id, related)

        done += len(cont_cfg["global_queries"])
        progress.progress(min(done / total_q, 1.0), text=f"Processed {cont_name}…")

    progress.empty()
    st.success("✅ Data refreshed!")
    st.cache_data.clear()

if st.session_state.get("fetch_all"):
    with st.spinner("Fetching and analysing all news…"):
        fetch_and_analyse_all()
    st.session_state["fetch_all"] = False

# ── Load display data ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_overview_data(from_d, to_d):
    db = get_db()
    return db.get_articles(
        from_date=str(from_d), to_date=str(to_d), limit=1000
    )

df = load_overview_data(g_from, g_to)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(f"""
<h1 style='text-align:center; font-size:2rem; margin-bottom:4px;'>
  {PAGE_ICON} {PAGE_TITLE}
</h1>
<p style='text-align:center; color:#9AA0B4; margin-top:0;'>
  Real-time NLP sentiment analysis of financial news across Asia, Europe &amp; North America
  &nbsp;|&nbsp; {str(g_from)} → {str(g_to)}
</p>
<hr style='border-color:#2D3153;'>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────

total = len(df)
pos_pct = (df["sentiment_label"] == "positive").sum() / max(total, 1) * 100
neg_pct = (df["sentiment_label"] == "negative").sum() / max(total, 1) * 100
neu_pct = (df["sentiment_label"] == "neutral" ).sum() / max(total, 1) * 100
avg_compound = df["compound_score"].mean() if "compound_score" in df.columns and total > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, value, label, delta_color="#2ECC71"):
    col.markdown(
        f"<div class='metric-card'>"
        f"<h2 style='color:{delta_color}'>{value}</h2>"
        f"<p>{label}</p></div>",
        unsafe_allow_html=True,
    )

kpi(c1, f"{total:,}",       "Total Articles",        "#4ECDC4")
kpi(c2, f"{pos_pct:.1f}%",  "📈 Positive",           "#2ECC71")
kpi(c3, f"{neg_pct:.1f}%",  "📉 Negative",           "#E74C3C")
kpi(c4, f"{neu_pct:.1f}%",  "➡️  Neutral",            "#95A5A6")
kpi(c5, f"{avg_compound:+.3f}", "Avg Compound Score", "#F39C12")

st.markdown("<br>", unsafe_allow_html=True)

# ── World map ─────────────────────────────────────────────────────────────────

st.markdown("<div class='section-header'>🗺️ Global Sentiment Heatmap</div>", unsafe_allow_html=True)

# Build per-country aggregation
country_iso = {}
for cont_cfg in CONTINENTS.values():
    for cname, ccfg in cont_cfg["countries"].items():
        country_iso[cname] = ccfg["iso3"]

if total > 0 and "country" in df.columns:
    cdf = df.dropna(subset=["sentiment_label"]).copy()
    cdf["iso3"] = cdf["country"].map(country_iso)
    cdf["compound_score"] = pd.to_numeric(cdf["compound_score"], errors="coerce").fillna(0)
    map_df = (
        cdf.dropna(subset=["iso3"])
           .groupby(["country", "iso3"], as_index=False)
           .agg(
               article_count=("title", "count"),
               avg_score=("compound_score", "mean"),
               positive_pct=("sentiment_label", lambda x: (x == "positive").mean() * 100),
           )
    )

    fig_map = px.choropleth(
        map_df,
        locations="iso3",
        color="avg_score",
        hover_name="country",
        hover_data={"article_count": True, "positive_pct": ":.1f", "avg_score": ":.3f"},
        color_continuous_scale=["#E74C3C", "#95A5A6", "#2ECC71"],
        range_color=[-1, 1],
        labels={"avg_score": "Sentiment Score", "article_count": "Articles"},
        title="",
    )
    fig_map.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        geo=dict(bgcolor="#0E1117", showframe=False, showcoastlines=True,
                 coastlinecolor="#2D3153", countrycolor="#2D3153",
                 landcolor="#1A1D2E", oceancolor="#0E1117"),
        coloraxis_colorbar=dict(title="Sentiment", tickfont=dict(color="#E8EAED")),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No data yet. Click **Fetch All Continent News** in the sidebar to populate.")

# ── Sentiment gauge + distribution ────────────────────────────────────────────

col_gauge, col_dist, col_cont = st.columns([1, 1.2, 1.5])

with col_gauge:
    st.markdown("<div class='section-header'>📊 Market Sentiment</div>", unsafe_allow_html=True)
    gauge_val = (avg_compound + 1) / 2 * 100  # 0 – 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_compound,
        number={"suffix": "", "font": {"size": 26, "color": "#E8EAED"}},
        delta={"reference": 0, "increasing": {"color": "#2ECC71"}, "decreasing": {"color": "#E74C3C"}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#9AA0B4"},
            "bar":  {"color": "#4ECDC4"},
            "steps": [
                {"range": [-1, -0.1], "color": "#4B1B1B"},
                {"range": [-0.1, 0.1], "color": "#2B2E40"},
                {"range": [0.1,  1.0], "color": "#1B4B2E"},
            ],
            "threshold": {"line": {"color": "#FF6B6B", "width": 3}, "thickness": 0.8, "value": 0},
        },
        title={"text": "Compound Score<br><span style='font-size:.7rem;color:#9AA0B4'>[-1 bearish → +1 bullish]</span>",
               "font": {"color": "#E8EAED"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        height=260, margin=dict(l=20, r=20, t=60, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_dist:
    st.markdown("<div class='section-header'>🥧 Sentiment Split</div>", unsafe_allow_html=True)
    if total > 0:
        pie_df = df["sentiment_label"].value_counts().reset_index()
        pie_df.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(
            pie_df, values="Count", names="Sentiment",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.55,
        )
        fig_pie.update_traces(textinfo="label+percent", textfont_size=13)
        fig_pie.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color="#E8EAED"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Fetch data first")

with col_cont:
    st.markdown("<div class='section-header'>🌐 By Continent</div>", unsafe_allow_html=True)
    if total > 0 and "continent" in df.columns:
        cont_df = (
            df.dropna(subset=["sentiment_label", "continent"])
              .groupby(["continent", "sentiment_label"])
              .size()
              .reset_index(name="count")
        )
        fig_bar = px.bar(
            cont_df, x="continent", y="count", color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="group",
            labels={"count": "Articles", "continent": "", "sentiment_label": "Sentiment"},
        )
        fig_bar.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=260,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickfont=dict(color="#E8EAED")),
            yaxis=dict(tickfont=dict(color="#E8EAED"), gridcolor="#2D3153"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Fetch data first")

# ── Recent news feed ──────────────────────────────────────────────────────────

st.markdown("<hr style='border-color:#2D3153;'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📰 Latest News</div>", unsafe_allow_html=True)

if total > 0:
    recent = df.dropna(subset=["title"]).head(30)
    cols = st.columns(3)
    for i, (_, row) in enumerate(recent.iterrows()):
        sentiment = row.get("sentiment_label", "neutral") or "neutral"
        badge_cls = f"badge-{sentiment}"
        emoji     = SENTIMENT_EMOJIS.get(sentiment, "➡️")
        title     = row.get("title", "")[:120]
        source    = row.get("source_name", "")
        pub       = str(row.get("published_at", ""))[:10]
        country   = row.get("country", "")
        url       = row.get("url", "#")
        pos_s     = row.get("positive_score", 0) or 0
        neg_s     = row.get("negative_score", 0) or 0
        neu_s     = row.get("neutral_score",  0) or 0

        with cols[i % 3]:
            st.markdown(f"""
<div class='news-card'>
  <span class='{badge_cls}'>{emoji} {sentiment.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{country} · {pub}</small><br><br>
  <a href='{url}' target='_blank' style='color:#E8EAED;text-decoration:none;font-weight:600;font-size:.92rem;'>
    {title}
  </a><br>
  <small style='color:#9AA0B4'>{source}</small><br>
  <small style='color:#9AA0B4'>📈{pos_s:.2f} &nbsp; 📉{neg_s:.2f} &nbsp; ➡{neu_s:.2f}</small>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style='background:#1A1D2E;border-radius:12px;padding:40px;text-align:center;'>
  <h3>No articles cached yet</h3>
  <p>Use the sidebar to fetch news and run sentiment analysis.</p>
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    st.caption("⏱️ Auto-refreshing every 15 minutes…")
    time.sleep(900)
    st.rerun()
