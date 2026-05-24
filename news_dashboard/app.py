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
    NEWS_API_KEY, HF_TOKEN, CONTINENTS, SENTIMENT_COLORS, SENTIMENT_EMOJIS,
    PAGE_TITLE, PAGE_ICON, DB_PATH, NSE_STOCKS_CSV,
    CACHE_EXPIRY_HOURS, MAX_ARTICLES_PER_QUERY, DEFAULT_LOOKBACK_DAYS,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card{background:#1A1D2E;border-radius:12px;padding:18px 22px;
             border:1px solid #2D3153;text-align:center;}
.metric-card h2{font-size:2.2rem;margin:4px 0;}
.metric-card p{font-size:.82rem;color:#9AA0B4;margin:0;}
.news-card{background:#1A1D2E;border-radius:10px;padding:14px 18px;
           border:1px solid #2D3153;margin-bottom:10px;}
.badge-positive{background:#1B4B2E;color:#2ECC71;border-radius:6px;
                padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-negative{background:#4B1B1B;color:#E74C3C;border-radius:6px;
                padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-neutral{background:#2B2E40;color:#95A5A6;border-radius:6px;
               padding:3px 10px;font-size:.78rem;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

@st.cache_resource(show_spinner=False)
def get_mapper():
    from src.stock_mapper import StockMapper
    from src.sp500_stocks import get_sp500_df
    nse = pd.read_csv(NSE_STOCKS_CSV) if os.path.exists(NSE_STOCKS_CSV) else pd.DataFrame()
    return StockMapper(nse, get_sp500_df())

@st.cache_resource(show_spinner=False)
def get_analyzer():
    from src.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer(hf_token=HF_TOKEN)

def get_fetcher():
    from src.news_fetcher import NewsAPIFetcher
    return NewsAPIFetcher(NEWS_API_KEY, get_db(), CACHE_EXPIRY_HOURS)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 FinNews Sentiment")
    st.markdown("---")

    today        = date.today()
    default_from = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    g_from = st.date_input("From", value=default_from, max_value=today, key="g_from")
    g_to   = st.date_input("To",   value=today,         max_value=today, key="g_to")

    st.markdown("---")
    model_choice = st.radio(
        "Sentiment model",
        ["FinBERT (HF API)", "VADER (fast, local)"],
        index=1,
        help="VADER is instant. FinBERT calls the HuggingFace cloud API — first call warms the model (~20 s).",
    )
    use_finbert = model_choice.startswith("FinBERT")

    n_articles = st.slider("Articles per query", 5, 50, MAX_ARTICLES_PER_QUERY, 5)

    st.markdown("---")
    fetch_btn = st.button(
        "🔄  Fetch & Analyse News",
        type="primary",
        use_container_width=True,
        help="Calls NewsAPI then runs sentiment analysis on each headline.",
    )
    if fetch_btn:
        st.session_state["fetch_all"] = True

    st.markdown("---")
    db_stats = get_db().stats()
    st.metric("Cached Articles",    db_stats["total_articles"])
    st.metric("Sentiment-Analysed", db_stats["analyzed"])
    st.metric("Stock-Mapped",       db_stats["stock_mapped"])
    st.caption("Data: [NewsAPI.org](https://newsapi.org)  ·  NLP: FinBERT / VADER")

# ── Fetch & analyse ───────────────────────────────────────────────────────────

def fetch_and_analyse_all():
    fetcher  = get_fetcher()
    analyzer = get_analyzer()
    mapper   = get_mapper()
    db       = get_db()
    model_key = "finbert" if use_finbert else "vader"

    all_errors: list[str] = []
    total_q = sum(len(cfg["global_queries"]) for cfg in CONTINENTS.values())
    progress = st.progress(0.0, text="Starting fetch…")
    done = 0

    for cont_name, cont_cfg in CONTINENTS.items():
        progress.progress(done / total_q, text=f"Fetching {cont_name} news…")
        articles, errors = fetcher.fetch_for_continent(
            cont_name, cont_cfg,
            from_date=g_from, to_date=g_to,
            max_per_query=n_articles,
        )
        all_errors.extend(errors)
        if errors:
            # Stop early on auth / quota errors
            break

        if articles:
            texts = [f"{a['title']} {a.get('description','')}" for a in articles]
            progress.progress(done / total_q, text=f"Analysing {cont_name} ({len(articles)} articles)…")

            if use_finbert:
                sentiments = analyzer.analyze_batch_finbert(texts)
            else:
                sentiments = analyzer.analyze_batch_vader(texts)

            for art, sent in zip(articles, sentiments):
                if not sent:
                    continue
                art_id = db.upsert_article(art)
                if art_id:
                    db.upsert_sentiment(art_id, model_key, sent)
                    related = mapper.find_related_stocks(art["title"], art.get("description", ""))
                    if related:
                        db.upsert_stock_mentions(art_id, related)

        done += len(cont_cfg["global_queries"])

    progress.empty()
    st.cache_data.clear()

    if all_errors:
        st.error(
            "**NewsAPI errors encountered:**\n\n" +
            "\n".join(f"- {e}" for e in all_errors[:5])
        )
    else:
        st.success(f"Done! Fetched & analysed news using **{model_key.upper()}**.")


if st.session_state.get("fetch_all"):
    with st.spinner("Working…"):
        fetch_and_analyse_all()
    st.session_state["fetch_all"] = False

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_overview_data(from_d, to_d):
    return get_db().get_articles(from_date=str(from_d), to_date=str(to_d), limit=1000)

df = load_overview_data(g_from, g_to)
total = len(df)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<h1 style='text-align:center;font-size:2rem;margin-bottom:4px;'>
  {PAGE_ICON} {PAGE_TITLE}
</h1>
<p style='text-align:center;color:#9AA0B4;margin-top:0;'>
  Real-time NLP sentiment of financial news &nbsp;|&nbsp;
  Asia &middot; Europe &middot; North America &nbsp;|&nbsp; NSE &amp; S&amp;P 500
  &nbsp;|&nbsp; {str(g_from)} &rarr; {str(g_to)}
</p>
<hr style='border-color:#2D3153;'>
""", unsafe_allow_html=True)

# ── Empty-state CTA ───────────────────────────────────────────────────────────

if total == 0:
    st.markdown("""
<div style='background:#1A1D2E;border-radius:14px;padding:36px 40px;
            border:1px solid #2D3153;text-align:center;margin:20px 0;'>
  <h2 style='margin-top:0;'>Welcome! No data yet.</h2>
  <p style='color:#9AA0B4;font-size:1.05rem;'>
    Click <strong style='color:#FF6B6B;'>Fetch &amp; Analyse News</strong> in the
    left sidebar to pull live headlines and run sentiment analysis.
  </p>
  <p style='color:#9AA0B4;font-size:.9rem;'>
    &bull; <b>VADER</b> mode is instant (recommended for first run)<br>
    &bull; <b>FinBERT</b> mode calls the HuggingFace API (~20 s warm-up on first use)<br>
    &bull; Results are cached locally — subsequent loads are instant
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────

pos_pct = (df["sentiment_label"] == "positive").sum() / total * 100
neg_pct = (df["sentiment_label"] == "negative").sum() / total * 100
neu_pct = (df["sentiment_label"] == "neutral" ).sum() / total * 100
avg_compound = (
    pd.to_numeric(df["compound_score"], errors="coerce").mean()
    if "compound_score" in df.columns else 0
)

c1, c2, c3, c4, c5 = st.columns(5)
def kpi(col, val, label, color="#4ECDC4"):
    col.markdown(
        f"<div class='metric-card'>"
        f"<h2 style='color:{color}'>{val}</h2><p>{label}</p></div>",
        unsafe_allow_html=True,
    )

kpi(c1, f"{total:,}",          "Total Articles",     "#4ECDC4")
kpi(c2, f"{pos_pct:.1f}%",     "📈 Positive",        "#2ECC71")
kpi(c3, f"{neg_pct:.1f}%",     "📉 Negative",        "#E74C3C")
kpi(c4, f"{neu_pct:.1f}%",     "➡️  Neutral",         "#95A5A6")
kpi(c5, f"{avg_compound:+.3f}","Avg Compound Score",  "#F39C12")
st.markdown("<br>", unsafe_allow_html=True)

# ── World map ─────────────────────────────────────────────────────────────────

st.markdown("### 🗺️ Global Sentiment Heatmap")

country_iso = {
    cname: ccfg["iso3"]
    for cfg in CONTINENTS.values()
    for cname, ccfg in cfg["countries"].items()
}

cdf = df.dropna(subset=["sentiment_label"]).copy()
cdf["iso3"]          = cdf["country"].map(country_iso)
cdf["compound_score"] = pd.to_numeric(cdf["compound_score"], errors="coerce").fillna(0)
map_df = (
    cdf.dropna(subset=["iso3"])
       .groupby(["country", "iso3"], as_index=False)
       .agg(
           article_count=("title",          "count"),
           avg_score    =("compound_score",  "mean"),
           positive_pct =("sentiment_label", lambda x: (x=="positive").mean()*100),
       )
)

if not map_df.empty:
    fig_map = px.choropleth(
        map_df, locations="iso3", color="avg_score",
        hover_name="country",
        hover_data={"article_count": True, "positive_pct": ":.1f", "avg_score": ":.3f"},
        color_continuous_scale=["#E74C3C", "#95A5A6", "#2ECC71"],
        range_color=[-1, 1],
        labels={"avg_score": "Sentiment Score", "article_count": "Articles"},
    )
    fig_map.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        geo=dict(bgcolor="#0E1117", showframe=False, showcoastlines=True,
                 coastlinecolor="#2D3153", countrycolor="#2D3153",
                 landcolor="#1A1D2E", oceancolor="#0E1117"),
        coloraxis_colorbar=dict(title="Sentiment", tickfont=dict(color="#E8EAED")),
        margin=dict(l=0,r=0,t=0,b=0), height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("World map needs articles with detected countries.")

# ── Gauge + Pie + Bar ─────────────────────────────────────────────────────────

col_g, col_p, col_b = st.columns([1, 1.2, 1.5])

with col_g:
    st.markdown("#### Market Sentiment Gauge")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_compound,
        number={"font": {"size": 26, "color": "#E8EAED"}},
        delta={"reference": 0,
               "increasing": {"color": "#2ECC71"},
               "decreasing": {"color": "#E74C3C"}},
        gauge={
            "axis":  {"range": [-1, 1], "tickcolor": "#9AA0B4"},
            "bar":   {"color": "#4ECDC4"},
            "steps": [
                {"range": [-1,   -0.1], "color": "#4B1B1B"},
                {"range": [-0.1,  0.1], "color": "#2B2E40"},
                {"range": [0.1,   1.0], "color": "#1B4B2E"},
            ],
            "threshold": {"line": {"color": "#FF6B6B", "width": 3},
                          "thickness": 0.8, "value": 0},
        },
        title={"text": "Compound Score<br><span style='font-size:.7rem;color:#9AA0B4'>[-1 bearish → +1 bullish]</span>",
               "font": {"color": "#E8EAED"}},
    ))
    fig_g.update_layout(paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                        height=260, margin=dict(l=20,r=20,t=60,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with col_p:
    st.markdown("#### Sentiment Split")
    pie = df["sentiment_label"].value_counts().reset_index()
    pie.columns = ["Sentiment", "Count"]
    fig_p = px.pie(pie, values="Count", names="Sentiment",
                   color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.55)
    fig_p.update_traces(textinfo="label+percent", textfont_size=13)
    fig_p.update_layout(paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                        showlegend=False, height=260,
                        margin=dict(l=10,r=10,t=10,b=10),
                        font=dict(color="#E8EAED"))
    st.plotly_chart(fig_p, use_container_width=True)

with col_b:
    st.markdown("#### By Continent")
    if "continent" in df.columns:
        bdf = (
            df.dropna(subset=["sentiment_label","continent"])
              .groupby(["continent","sentiment_label"])
              .size().reset_index(name="count")
        )
        fig_b = px.bar(bdf, x="continent", y="count", color="sentiment_label",
                       color_discrete_map=SENTIMENT_COLORS, barmode="group",
                       labels={"count":"Articles","continent":"","sentiment_label":"Sentiment"})
        fig_b.update_layout(paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                            font=dict(color="#E8EAED"), height=260,
                            legend=dict(orientation="h",y=-0.25),
                            margin=dict(l=10,r=10,t=10,b=10),
                            yaxis=dict(gridcolor="#2D3153"))
        st.plotly_chart(fig_b, use_container_width=True)

# ── News feed ─────────────────────────────────────────────────────────────────

st.markdown("<hr style='border-color:#2D3153;'>", unsafe_allow_html=True)
st.markdown("### 📰 Latest Headlines")

recent = df.dropna(subset=["title"]).head(30)
cols   = st.columns(3)
def _safe_label(val) -> str:
    return val if isinstance(val, str) and val in ("positive", "negative", "neutral") else "neutral"

for i, (_, row) in enumerate(recent.iterrows()):
    sentiment = _safe_label(row.get("sentiment_label"))
    badge_cls = f"badge-{sentiment}"
    emoji     = SENTIMENT_EMOJIS.get(sentiment, "➡️")
    title     = (row.get("title") or "")[:120]
    source    = row.get("source_name") or ""
    pub       = str(row.get("published_at") or "")[:10]
    country   = row.get("country") or ""
    url       = row.get("url")     or "#"
    pos_s     = float(row.get("positive_score") or 0)
    neg_s     = float(row.get("negative_score") or 0)
    neu_s     = float(row.get("neutral_score")  or 0)

    with cols[i % 3]:
        st.markdown(f"""
<div class='news-card'>
  <span class='{badge_cls}'>{emoji} {sentiment.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{country} · {pub}</small><br><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:600;font-size:.92rem;'>
    {title}
  </a><br>
  <small style='color:#9AA0B4'>{source}</small><br>
  <small style='color:#9AA0B4'>
    📈{pos_s:.2f} &nbsp; 📉{neg_s:.2f} &nbsp; ➡{neu_s:.2f}
  </small>
</div>
""", unsafe_allow_html=True)
