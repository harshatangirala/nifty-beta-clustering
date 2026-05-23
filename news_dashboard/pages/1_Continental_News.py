"""Page 1 – Continental News Browser"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from config import (
    NEWS_API_KEY, HF_TOKEN, CONTINENTS, SENTIMENT_COLORS, SENTIMENT_EMOJIS,
    DB_PATH, CACHE_EXPIRY_HOURS, MAX_ARTICLES_PER_QUERY,
)

st.set_page_config(page_title="Continental News", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.news-card { background:#1A1D2E; border-radius:10px; padding:14px 18px;
             border:1px solid #2D3153; margin-bottom:10px; }
.badge-positive { background:#1B4B2E; color:#2ECC71; border-radius:6px;
                  padding:3px 10px; font-size:.78rem; font-weight:700; }
.badge-negative { background:#4B1B1B; color:#E74C3C; border-radius:6px;
                  padding:3px 10px; font-size:.78rem; font-weight:700; }
.badge-neutral  { background:#2B2E40; color:#95A5A6; border-radius:6px;
                  padding:3px 10px; font-size:.78rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    from src.database import NewsDatabase
    return NewsDatabase(DB_PATH)

@st.cache_resource
def get_analyzer():
    from src.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer(hf_token=HF_TOKEN)

@st.cache_resource
def get_mapper():
    from src.stock_mapper import StockMapper
    from src.sp500_stocks import get_sp500_df
    nse = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "master_stock_universe.csv"))
    return StockMapper(nse, get_sp500_df())

# ── Sidebar filters ───────────────────────────────────────────────────────────

st.sidebar.title("🌍 Continental News")

continent_choice = st.sidebar.selectbox(
    "Continent", list(CONTINENTS.keys()), key="cont_sel"
)
cont_cfg = CONTINENTS[continent_choice]

country_list = ["All Countries"] + list(cont_cfg["countries"].keys())
country_choice = st.sidebar.selectbox("Country", country_list, key="country_sel")

st.sidebar.markdown("---")
today       = date.today()
from_date   = st.sidebar.date_input("From", today - timedelta(days=7), max_value=today)
to_date     = st.sidebar.date_input("To",   today,                     max_value=today)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment", ["positive", "negative", "neutral"],
    default=["positive", "negative", "neutral"],
)
n_articles = st.sidebar.slider("Max articles", 5, 100, 50, 5)

st.sidebar.markdown("---")
use_finbert = st.sidebar.checkbox("Use FinBERT", value=True)

if st.sidebar.button("🔄 Fetch & Analyse", type="primary", use_container_width=True):
    st.session_state["do_fetch_cont"] = True

# ── Fetch logic ───────────────────────────────────────────────────────────────

def fetch_and_analyse(continent_name, country_name, from_d, to_d, max_n):
    from src.news_fetcher import NewsAPIFetcher
    fetcher  = NewsAPIFetcher(NEWS_API_KEY, get_db(), CACHE_EXPIRY_HOURS)
    analyzer = get_analyzer()
    mapper   = get_mapper()
    db       = get_db()

    if country_name and country_name != "All Countries":
        ccfg                = cont_cfg["countries"][country_name]
        articles, errors    = fetcher.fetch_for_country(
            country_name, ccfg, continent_name,
            from_date=from_d, to_date=to_d, max_per_query=max_n,
        )
    else:
        articles, errors = fetcher.fetch_for_continent(
            continent_name, cont_cfg,
            from_date=from_d, to_date=to_d, max_per_query=max_n,
        )

    if errors:
        st.error("**NewsAPI error:** " + " | ".join(errors[:3]))

    if not articles:
        return 0

    texts     = [f"{a['title']} {a.get('description','')}" for a in articles]
    model_key = "finbert" if (use_finbert and analyzer.finbert_available()) else "vader"

    with st.spinner(f"Running {model_key.upper()} on {len(articles)} articles…"):
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
            related = mapper.find_related_stocks(art["title"], art.get("description",""))
            if related:
                db.upsert_stock_mentions(art_id, related)

    st.cache_data.clear()
    return len(articles)

if st.session_state.get("do_fetch_cont"):
    with st.spinner("Fetching news…"):
        n = fetch_and_analyse(
            continent_choice,
            country_choice if country_choice != "All Countries" else None,
            from_date, to_date, n_articles,
        )
    st.success(f"Fetched {n} articles")
    st.session_state["do_fetch_cont"] = False

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_cont_data(cont, country, from_d, to_d, sent_filter):
    db = get_db()
    dfs = []
    for s in sent_filter:
        _df = db.get_articles(
            from_date=str(from_d), to_date=str(to_d),
            continent=cont,
            country=None if country == "All Countries" else country,
            sentiment=s,
        )
        dfs.append(_df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df = load_cont_data(
    continent_choice, country_choice, from_date, to_date, sentiment_filter
)

# ── Page header ───────────────────────────────────────────────────────────────

icon = cont_cfg["icon"]
st.markdown(
    f"<h2>{icon} {continent_choice} Financial News Sentiment</h2>"
    f"<p style='color:#9AA0B4'>{str(from_date)} → {str(to_date)} "
    f"· {len(df)} articles · "
    f"{'All Countries' if country_choice == 'All Countries' else country_choice}</p>",
    unsafe_allow_html=True,
)

# ── Summary stats ─────────────────────────────────────────────────────────────

if len(df) > 0:
    total = len(df)
    pos_n = (df["sentiment_label"] == "positive").sum()
    neg_n = (df["sentiment_label"] == "negative").sum()
    neu_n = (df["sentiment_label"] == "neutral" ).sum()
    avg_c = df["compound_score"].mean() if "compound_score" in df else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Articles", total)
    c2.metric("Positive 📈", f"{pos_n} ({pos_n/total*100:.0f}%)")
    c3.metric("Negative 📉", f"{neg_n} ({neg_n/total*100:.0f}%)")
    c4.metric("Neutral ➡️",  f"{neu_n} ({neu_n/total*100:.0f}%)")
    c5.metric("Avg Compound", f"{avg_c:+.3f}")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────

    col_bar, col_country = st.columns(2)

    with col_bar:
        st.subheader("Sentiment Distribution")
        dist = df["sentiment_label"].value_counts().reset_index()
        dist.columns = ["Sentiment", "Count"]
        fig = px.bar(
            dist, x="Sentiment", y="Count", color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), showlegend=False,
            yaxis=dict(gridcolor="#2D3153"),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_country:
        st.subheader("By Country")
        if "country" in df.columns:
            cdf = (
                df.dropna(subset=["country", "sentiment_label"])
                  .groupby(["country", "sentiment_label"])
                  .size().reset_index(name="count")
            )
            fig2 = px.bar(
                cdf, x="country", y="count", color="sentiment_label",
                color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                labels={"count": "Articles", "country": "", "sentiment_label": "Sentiment"},
            )
            fig2.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font=dict(color="#E8EAED"), height=280,
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#2D3153"),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Sentiment timeline ────────────────────────────────────────────────────
    st.subheader("📅 Sentiment Over Time")
    if "published_at" in df.columns:
        tdf = df.copy()
        tdf["date"] = pd.to_datetime(tdf["published_at"], errors="coerce").dt.date
        tdf = tdf.dropna(subset=["date", "sentiment_label"])
        daily = (
            tdf.groupby(["date", "sentiment_label"])
               .size().reset_index(name="count")
        )
        fig3 = px.line(
            daily, x="date", y="count", color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS,
            markers=True,
            labels={"count": "Articles", "date": "Date", "sentiment_label": "Sentiment"},
        )
        fig3.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font=dict(color="#E8EAED"), height=280,
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(gridcolor="#2D3153"),
            yaxis=dict(gridcolor="#2D3153"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Score distribution histograms ─────────────────────────────────────────
    with st.expander("📊 Probability Score Distributions"):
        hist_cols = st.columns(3)
        for col, field, label, color in zip(
            hist_cols,
            ["positive_score", "negative_score", "neutral_score"],
            ["Positive Prob.", "Negative Prob.", "Neutral Prob."],
            ["#2ECC71", "#E74C3C", "#95A5A6"],
        ):
            if field in df.columns:
                fig_h = px.histogram(
                    df.dropna(subset=[field]), x=field,
                    nbins=20, color_discrete_sequence=[color],
                    labels={field: label},
                    title=label,
                )
                fig_h.update_layout(
                    paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                    font=dict(color="#E8EAED"), height=220,
                    showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(gridcolor="#2D3153"),
                )
                col.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")

    # ── News cards grid ───────────────────────────────────────────────────────
    st.subheader(f"📰 Articles ({min(len(df), n_articles)} shown)")

    cols = st.columns(2)
    shown = df.dropna(subset=["title"]).head(n_articles)
    for i, (_, row) in enumerate(shown.iterrows()):
        sentiment = row.get("sentiment_label", "neutral") or "neutral"
        badge_cls = f"badge-{sentiment}"
        emoji     = SENTIMENT_EMOJIS.get(sentiment, "➡️")
        title     = row.get("title", "")[:140]
        source    = row.get("source_name", "")
        pub       = str(row.get("published_at", ""))[:16].replace("T", " ")
        country   = row.get("country", "")
        url       = row.get("url", "#")
        desc      = (row.get("description", "") or "")[:200]
        pos_s     = float(row.get("positive_score", 0) or 0)
        neg_s     = float(row.get("negative_score", 0) or 0)
        neu_s     = float(row.get("neutral_score",  0) or 0)
        compound  = float(row.get("compound_score",  0) or 0)

        with cols[i % 2]:
            st.markdown(f"""
<div class='news-card'>
  <span class='{badge_cls}'>{emoji} {sentiment.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>🌍 {country}  ·  {pub}  ·  {source}</small><br><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:700;font-size:.94rem;'>
    {title}
  </a><br>
  <small style='color:#9AA0B4;display:block;margin-top:6px;'>{desc}</small><br>
  <small>
    <span style='color:#2ECC71'>📈 {pos_s:.2f}</span>&nbsp;
    <span style='color:#E74C3C'>📉 {neg_s:.2f}</span>&nbsp;
    <span style='color:#95A5A6'>➡ {neu_s:.2f}</span>&nbsp;
    <span style='color:#F39C12'>⚡ {compound:+.3f}</span>
  </small>
</div>
""", unsafe_allow_html=True)

else:
    st.info(
        "No cached articles found for the selected filters. "
        "Click **Fetch & Analyse** in the sidebar."
    )
