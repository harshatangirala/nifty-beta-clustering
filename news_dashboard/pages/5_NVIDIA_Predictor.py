"""
Page 5 – NVIDIA Stock Price Prediction Dashboard
=================================================
Pipeline:
  1. NVDA OHLCV  – Yahoo Finance (2015-01-01 → user end date)
  2. NVIDIA news – NewsAPI.org (last N days)
  3. Sentiment   – VADER (local) or FinBERT (HF Inference API)
  4. Cumulative sentiment score
  5. Sentiment model evaluation on Financial Phrase Bank
  6. Ridge-regression models (price-only + sentiment-augmented)
  7. Monte-Carlo / GBM price simulation → probability distributions

All heavy results are stored in st.session_state so they survive sidebar
interactions without disappearing.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import warnings, requests, time, traceback
warnings.filterwarnings("ignore")

from scipy.stats import norm, shapiro
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWS_API_KEY, HF_TOKEN

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NVIDIA Price Predictor",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.nvda-header{background:linear-gradient(135deg,#1a1d2e 0%,#0d1f0d 100%);
             border-radius:16px;padding:28px 36px;border:1px solid #2a5c2a;
             margin-bottom:18px;}
.nvda-header h1{color:#76B900;margin:0;font-size:2rem;}
.nvda-header p{color:#9AA0B4;margin:6px 0 0 0;font-size:.95rem;}
.metric-card{background:#1A1D2E;border-radius:12px;padding:18px 22px;
             border:1px solid #2D3153;text-align:center;}
.metric-card h2{font-size:1.85rem;margin:4px 0;}
.metric-card p{font-size:.78rem;color:#9AA0B4;margin:0;}
.section-box{background:#1A1D2E;border-radius:12px;padding:18px 22px;
             border:1px solid #2D3153;margin-bottom:14px;font-size:.92rem;}
.news-card{background:#141726;border-radius:10px;padding:12px 16px;
           border:1px solid #2D3153;margin-bottom:9px;}
.badge-positive{background:#1B4B2E;color:#2ECC71;border-radius:6px;
                padding:2px 9px;font-size:.76rem;font-weight:700;}
.badge-negative{background:#4B1B1B;color:#E74C3C;border-radius:6px;
                padding:2px 9px;font-size:.76rem;font-weight:700;}
.badge-neutral{background:#2B2E40;color:#95A5A6;border-radius:6px;
               padding:2px 9px;font-size:.76rem;font-weight:700;}
.info-tag{background:#1A2D3E;color:#4ECDC4;border-radius:6px;
          padding:2px 9px;font-size:.76rem;font-weight:600;}
.warn-box{background:#2B2000;border:1px solid #F39C12;border-radius:8px;
          padding:10px 16px;color:#F39C12;font-size:.87rem;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

TICKER      = "NVDA"
TRAIN_START = "2015-01-01"
CHART_BG    = "#0E1117"
CARD_BG     = "#1A1D2E"
GRID_COLOR  = "#2D3153"

NVIDIA_QUERIES = [
    "NVIDIA earnings revenue profit",
    "NVIDIA GPU AI chip data center",
    "NVIDIA Jensen Huang CEO",
    "NVIDIA Blackwell Hopper semiconductor",
    "NVDA stock analyst price target",
    "NVIDIA supply chain demand outlook",
]

COLORS = {
    "nvda":     "#76B900",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral":  "#95A5A6",
    "predict":  "#F39C12",
    "ci_lo":    "#FF6B6B",
    "ci_hi":    "#4ECDC4",
    "blue":     "#45B7D1",
}

_LAYOUT = dict(
    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    font=dict(color="#E8EAED", size=12),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

# ── Session-state initialisation ──────────────────────────────────────────────
# All pipeline results live here so they survive sidebar interactions.

_SS_DEFAULTS = {
    "nvda_news_sent":  None,   # DataFrame: articles + sentiment columns
    "nvda_daily_sent": None,   # DataFrame: daily aggregated sentiment
    "nvda_models":     None,   # dict: model results from train_price_models
    "nvda_feat_df":    None,   # DataFrame: price + technical features
    "nvda_mc":         None,   # dict: Monte Carlo output
    "nvda_mc_params":  None,   # tuple (pred_days, n_sims) when MC last ran
    "nvda_has_s":      False,  # bool: did training use sentiment?
    "run_eval":        False,
    "eval_results":    None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🟢 NVIDIA Predictor")
    st.markdown("---")

    end_date = st.date_input(
        "Price data end date",
        value=date.today(),
        max_value=date.today(),
        help="OHLCV fetched from 2015-01-01 to this date.",
    )
    pred_days = st.slider("Forecast horizon (days)", 5, 60, 30, 5)
    n_sims    = st.select_slider(
        "Monte Carlo simulations",
        options=[1_000, 5_000, 10_000, 50_000],
        value=10_000,
    )
    st.markdown("---")

    news_lookback = st.slider("News lookback (days)", 7, 30, 20,
                               help="NewsAPI free tier: ~30 days max.")
    max_articles  = st.slider("Max articles per query", 10, 100, 50, 10)
    model_choice  = st.radio(
        "Sentiment model",
        ["VADER (fast, local)", "FinBERT (HF API, slower)"],
        index=0,
    )
    use_finbert = model_choice.startswith("FinBERT")

    st.markdown("---")
    fetch_btn = st.button(
        "🔄  Fetch Data & Train",
        type="primary",
        use_container_width=True,
    )
    eval_btn = st.button(
        "📊  Evaluate Sentiment Models",
        use_container_width=True,
        help="Runs VADER (and optionally FinBERT) on 45 Financial Phrase Bank samples.",
    )
    if eval_btn:
        st.session_state["run_eval"] = True

    st.markdown("---")
    # Show pipeline status
    has_news   = st.session_state["nvda_news_sent"] is not None
    has_models = st.session_state["nvda_models"]   is not None
    has_mc     = st.session_state["nvda_mc"]       is not None
    st.caption(
        f"News: {'✅' if has_news else '⬜'}  "
        f"Models: {'✅' if has_models else '⬜'}  "
        f"MC: {'✅' if has_mc else '⬜'}"
    )
    st.caption("Data: Yahoo Finance · NewsAPI.org")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class='nvda-header'>
  <h1>🟢 NVIDIA (NVDA) Price Prediction Dashboard</h1>
  <p>
    Sentiment-augmented Ridge regression &nbsp;|&nbsp; Monte Carlo GBM &nbsp;|&nbsp;
    Price: Yahoo Finance (2015 → {end_date}) &nbsp;|&nbsp; News: NewsAPI.org
  </p>
</div>
""", unsafe_allow_html=True)

# ── Cached data-fetching helpers ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(end_date_str: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(
            TICKER, start=TRAIN_START, end=end_date_str,
            interval="1d", auto_adjust=True, progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        raw.index.name = "Date"
        df = raw.reset_index()
        # Flatten MultiIndex columns (field, ticker) → field only
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        _dt = pd.to_datetime(df["date"])
        df["date"] = (_dt.dt.tz_convert(None) if _dt.dt.tz is not None else _dt).dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        needed = [c for c in ["date", "open", "high", "low", "close", "volume"]
                  if c in df.columns]
        return df[needed].dropna().sort_values("date").reset_index(drop=True)
    except ImportError:
        st.error("yfinance not installed — run: pip install yfinance")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Price data fetch error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nvidia_news(api_key: str, lookback_days: int, max_per_q: int) -> pd.DataFrame:
    news_to   = date.today()
    news_from = news_to - timedelta(days=lookback_days)
    articles, seen = [], set()
    for query in NVIDIA_QUERIES:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query, "from": str(news_from), "to": str(news_to),
                    "language": "en", "sortBy": "publishedAt",
                    "pageSize": min(max_per_q, 100), "apiKey": api_key,
                },
                timeout=15,
            )
            if r.status_code != 200:
                continue
            for art in r.json().get("articles", []):
                url = art.get("url", "")
                if url in seen or not art.get("title"):
                    continue
                seen.add(url)
                articles.append({
                    "title":        art.get("title", ""),
                    "description":  art.get("description") or "",
                    "url":          url,
                    "source":       (art.get("source") or {}).get("name", ""),
                    "published_at": art.get("publishedAt", ""),
                })
            time.sleep(0.12)
        except Exception:
            continue
    if not articles:
        return pd.DataFrame()
    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["published_at"].dt.tz_convert(None).dt.normalize()
    return df.dropna(subset=["date", "title"]).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def run_vader_cached(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    sid  = SentimentIntensityAnalyzer()
    rows = []
    for _, row in news_df.iterrows():
        text = f"{row['title']} {row.get('description', '')}"[:512]
        sc   = sid.polarity_scores(text)
        c    = sc["compound"]
        rows.append({
            "compound": round(c, 4),
            "positive": round(sc["pos"], 4),
            "negative": round(sc["neg"], 4),
            "neutral":  round(sc["neu"], 4),
            "label":    "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"),
        })
    return news_df.assign(**pd.DataFrame(rows))


@st.cache_data(ttl=3600, show_spinner=False)
def run_finbert_cached(news_df: pd.DataFrame, hf_token: str) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    hdrs   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    texts  = [f"{r['title']} {r.get('description', '')}"[:512]
              for _, r in news_df.iterrows()]
    rows   = []
    for i in range(0, len(texts), 8):
        batch = texts[i: i + 8]
        done  = False
        try:
            resp = requests.post(
                HF_URL, headers=hdrs,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=90,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    items = item if isinstance(item, list) else [item]
                    sc  = {r["label"].lower(): r["score"] for r in items}
                    pos = sc.get("positive", 0)
                    neg = sc.get("negative", 0)
                    c   = round(pos - neg, 4)
                    rows.append({
                        "compound": c,
                        "positive": round(pos, 4),
                        "negative": round(neg, 4),
                        "neutral":  round(sc.get("neutral", 0), 4),
                        "label":    "positive" if c >= 0.05 else (
                                    "negative" if c <= -0.05 else "neutral"),
                    })
                done = True
        except Exception:
            pass
        if not done:
            rows.extend([{"compound": 0, "positive": 0,
                           "negative": 0, "neutral": 1, "label": "neutral"}] * len(batch))
        time.sleep(0.15)
    # Pad if short
    rows.extend([{"compound": 0, "positive": 0,
                   "negative": 0, "neutral": 1, "label": "neutral"}]
                 * max(0, len(news_df) - len(rows)))
    return news_df.assign(**pd.DataFrame(rows[:len(news_df)]))


# ── Non-cached helpers ────────────────────────────────────────────────────────

def compute_daily_sentiment(news_sent: pd.DataFrame) -> pd.DataFrame:
    if news_sent.empty:
        return pd.DataFrame()
    df  = news_sent.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = (
        df.groupby("date")
          .agg(
              avg_compound   = ("compound", "mean"),
              total_articles = ("compound", "count"),
              pos_count      = ("label",    lambda x: (x == "positive").sum()),
              neg_count      = ("label",    lambda x: (x == "negative").sum()),
              neu_count      = ("label",    lambda x: (x == "neutral").sum()),
          )
          .reset_index()
          .sort_values("date")
    )
    out["cumulative_compound"] = out["avg_compound"].cumsum()
    out["compound_ma3"]        = out["avg_compound"].rolling(3, min_periods=1).mean()
    out["compound_ma5"]        = out["avg_compound"].rolling(5, min_periods=1).mean()
    out["pos_ratio"]           = out["pos_count"] / out["total_articles"]
    out["neg_ratio"]           = out["neg_count"] / out["total_articles"]
    return out


def _kpi(col, val, lbl, color="#4ECDC4"):
    col.markdown(
        f"<div class='metric-card'><h2 style='color:{color}'>{val}</h2>"
        f"<p>{lbl}</p></div>",
        unsafe_allow_html=True,
    )


# ── Full pipeline (runs on button press, stores to session state) ─────────────

def run_full_pipeline(price_df: pd.DataFrame):
    """Fetch news → sentiment → train models → Monte Carlo. Stores everything
    in st.session_state so results survive sidebar interactions."""

    prog = st.progress(0.0, text="Starting pipeline…")
    try:
        # ── 1. Fetch news ─────────────────────────────────────────────────
        prog.progress(0.05, text="Fetching NVIDIA news from NewsAPI…")
        fetch_nvidia_news.clear()
        run_vader_cached.clear()
        run_finbert_cached.clear()
        news_df = fetch_nvidia_news(NEWS_API_KEY, news_lookback, max_articles)

        if news_df.empty:
            st.warning("No news articles returned. Check the API key or try a longer lookback.")
            st.session_state["nvda_news_sent"]  = pd.DataFrame()
            st.session_state["nvda_daily_sent"] = pd.DataFrame()
        else:
            # ── 2. Sentiment analysis ─────────────────────────────────────
            model_lbl = "FinBERT" if use_finbert else "VADER"
            prog.progress(0.20, text=f"Running {model_lbl} on {len(news_df)} articles…")
            if use_finbert:
                news_sent = run_finbert_cached(news_df, HF_TOKEN)
            else:
                news_sent = run_vader_cached(news_df)
            daily_sent = compute_daily_sentiment(news_sent)
            st.session_state["nvda_news_sent"]  = news_sent
            st.session_state["nvda_daily_sent"] = daily_sent

        # ── 3. Feature engineering + training ────────────────────────────
        prog.progress(0.45, text="Engineering features and training Ridge regression…")
        from src.nvidia_predictor import (
            compute_technical_features, merge_sentiment,
            train_price_models, SENTIMENT_FEATURES,
        )
        feat_df  = compute_technical_features(price_df)
        ds       = st.session_state["nvda_daily_sent"]
        has_s    = ds is not None and not ds.empty
        if has_s:
            feat_df = merge_sentiment(feat_df, ds)
        else:
            for c in SENTIMENT_FEATURES:
                feat_df[c] = 0.0
        models = train_price_models(feat_df, has_sentiment=has_s)

        st.session_state["nvda_feat_df"] = feat_df
        st.session_state["nvda_models"]  = models
        st.session_state["nvda_has_s"]   = has_s

        # ── 4. Monte Carlo ────────────────────────────────────────────────
        prog.progress(0.75, text=f"Monte Carlo: {n_sims:,} paths × {pred_days} days…")
        if models:
            from src.nvidia_predictor import monte_carlo_forecast
            active_key = (
                "sentiment_model"
                if (has_s and "sentiment_model" in models)
                else "price_model"
            )
            if active_key in models:
                mc = monte_carlo_forecast(
                    models[active_key], price_df, feat_df,
                    n_days=pred_days, n_simulations=n_sims,
                )
                st.session_state["nvda_mc"]        = mc
                st.session_state["nvda_mc_params"] = (pred_days, n_sims)

        prog.progress(1.0, text="Done!")
        prog.empty()
        n_art = len(st.session_state["nvda_news_sent"]) \
                if st.session_state["nvda_news_sent"] is not None else 0
        st.success(
            f"✅ Pipeline complete — {n_art} articles analysed, "
            f"models trained, {n_sims:,} MC paths generated."
        )
    except Exception:
        prog.empty()
        st.error("Pipeline failed — see traceback below.")
        st.code(traceback.format_exc())


# ── Trigger pipeline when button pressed ─────────────────────────────────────

if fetch_btn:
    run_full_pipeline(fetch_price_data(str(end_date)))

# ── Regenerate MC if forecast params changed since last run ───────────────────

mc_params = st.session_state["nvda_mc_params"]
if (
    st.session_state["nvda_mc"] is not None
    and st.session_state["nvda_models"] is not None
    and mc_params != (pred_days, n_sims)
):
    from src.nvidia_predictor import monte_carlo_forecast
    _models  = st.session_state["nvda_models"]
    _feat_df = st.session_state["nvda_feat_df"]
    _price   = fetch_price_data(str(end_date))
    _has_s   = st.session_state["nvda_has_s"]
    _ak      = "sentiment_model" if (_has_s and "sentiment_model" in _models) else "price_model"
    if _ak in _models and _price is not None and not _price.empty:
        with st.spinner("Re-running Monte Carlo with updated parameters…"):
            _mc = monte_carlo_forecast(
                _models[_ak], _price, _feat_df,
                n_days=pred_days, n_simulations=n_sims,
            )
        st.session_state["nvda_mc"]        = _mc
        st.session_state["nvda_mc_params"] = (pred_days, n_sims)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_candlestick(price_df: pd.DataFrame, end_lbl: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=price_df["date"], open=price_df["open"], high=price_df["high"],
        low=price_df["low"], close=price_df["close"],
        increasing_line_color=COLORS["nvda"],
        decreasing_line_color=COLORS["negative"],
        name="NVDA", showlegend=False,
    ), row=1, col=1)
    for w, col in [(50, "#F39C12"), (200, "#4ECDC4")]:
        fig.add_trace(go.Scatter(
            x=price_df["date"],
            y=price_df["close"].rolling(w).mean(),
            name=f"SMA{w}", line=dict(color=col, width=1.2, dash="dot"),
        ), row=1, col=1)
    colours = [
        COLORS["nvda"] if c >= o else COLORS["negative"]
        for c, o in zip(price_df["close"], price_df["open"])
    ]
    fig.add_trace(go.Bar(
        x=price_df["date"], y=price_df["volume"],
        name="Volume", marker_color=colours, opacity=0.6, showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        **_LAYOUT, height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        title=dict(
            text=f"<b>NVDA</b> Historical Price  (2015 → {end_lbl})",
            font=dict(size=14, color="#E8EAED"),
        ),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)
    return fig


def plot_cumulative_sentiment(daily_s: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.48], vertical_spacing=0.05,
        subplot_titles=["Cumulative News Score", "Daily Avg Compound Score"],
    )
    fig.add_trace(go.Scatter(
        x=daily_s["date"], y=daily_s["cumulative_compound"],
        fill="tozeroy", fillcolor="rgba(118,185,0,0.15)",
        line=dict(color=COLORS["nvda"], width=2), name="Cumulative",
    ), row=1, col=1)
    bar_cols = [
        COLORS["positive"] if v > 0.05 else (
        COLORS["negative"] if v < -0.05 else COLORS["neutral"])
        for v in daily_s["avg_compound"]
    ]
    fig.add_trace(go.Bar(
        x=daily_s["date"], y=daily_s["avg_compound"],
        marker_color=bar_cols, opacity=0.7, name="Daily",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=daily_s["date"], y=daily_s["compound_ma5"],
        line=dict(color=COLORS["predict"], width=2), name="5-Day MA",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1, row=2, col=1)
    fig.update_layout(**_LAYOUT, height=400, legend=dict(orientation="h", y=-0.08))
    return fig


def plot_article_scatter(news_s: pd.DataFrame) -> go.Figure:
    df = news_s.copy()
    df["short_title"] = df["title"].str[:60] + "…"
    fig = px.scatter(
        df, x="date", y="compound", color="label",
        color_discrete_map={k: COLORS[k] for k in ("positive", "negative", "neutral")},
        hover_data={"short_title": True, "source": True, "compound": ":.3f"},
        labels={"compound": "Compound", "date": "Date", "label": "Sentiment"},
    )
    for y_val, col, lbl in [
        ( 0.05, COLORS["positive"], "+0.05 threshold"),
        (-0.05, COLORS["negative"], "−0.05 threshold"),
        ( 0,    "#9AA0B4",          ""),
    ]:
        fig.add_hline(y=y_val, line_dash="dot" if y_val else "dash",
                      line_color=col, line_width=1,
                      annotation_text=lbl if lbl else "")
    fig.update_layout(
        **_LAYOUT, height=300,
        title=dict(text="<b>Article-Level Sentiment Scores</b>", font=dict(size=13)),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def plot_model_perf(mr: dict, title: str) -> go.Figure:
    dates  = pd.to_datetime(mr["test_dates"])
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=mr["test_price_actual"],
        name="Actual", line=dict(color=COLORS["nvda"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=mr["test_price_predicted"],
        name="Predicted", line=dict(color=COLORS["predict"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **_LAYOUT, height=330,
        title=dict(text=f"<b>Actual vs Predicted Price — {title}</b>",
                   font=dict(size=13)),
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def plot_residuals(residuals: np.ndarray) -> go.Figure:
    mu, std = float(np.mean(residuals)), float(np.std(residuals))
    x_rng   = np.linspace(mu - 4 * std, mu + 4 * std, 200)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Distribution", "Over Time"])
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, marker_color=COLORS["blue"],
        opacity=0.65, histnorm="probability density", name="Residuals",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_rng, y=norm.pdf(x_rng, mu, std),
        line=dict(color=COLORS["predict"], width=2), name="Normal fit",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))), y=list(residuals),
        mode="markers",
        marker=dict(color=COLORS["blue"], size=3, opacity=0.5),
        name="Residual",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", row=1, col=2)
    fig.update_layout(**_LAYOUT, height=300, showlegend=False)
    return fig


def plot_feat_importance(feat_imp: pd.Series, title: str) -> go.Figure:
    from src.nvidia_predictor import FEATURE_LABELS
    top    = feat_imp.head(12)
    labels = [FEATURE_LABELS.get(f, f) for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=labels[::-1],
        orientation="h",
        marker=dict(
            color=top.values[::-1],
            colorscale=[[0, "#2D3153"], [0.5, "#45B7D1"], [1, "#76B900"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **_LAYOUT, height=360,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13)),
        xaxis_title="Abs. coefficient (scaled)",
        margin=dict(l=130, r=20, t=50, b=20),
    )
    return fig


def plot_conf_matrix(cm: list, labels: list, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, CARD_BG], [0.5, "#1B4B8A"], [1, COLORS["nvda"]]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=14, color="#E8EAED"),
        showscale=False,
        hovertemplate="Pred: %{x}<br>True: %{y}<br>N: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT, height=270,
        title=dict(text=f"<b>{title} Confusion Matrix</b>", font=dict(size=13)),
        xaxis=dict(title="Predicted", gridcolor="transparent"),
        yaxis=dict(title="Actual",    gridcolor="transparent"),
        margin=dict(l=80, r=20, t=50, b=60),
    )
    return fig


def plot_prob_dist(mc: dict, horizon: int) -> go.Figure:
    pa       = mc["price_at"][horizon]
    curr     = mc["current_price"]
    mu, std  = pa["mean"], pa["std"]
    sim_px   = mc["paths"][:, horizon - 1]
    x_rng    = np.linspace(max(0.0, mu - 4 * std), mu + 4 * std, 400)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_px, nbinsx=80, name="Simulated prices",
        marker_color=COLORS["blue"], opacity=0.45,
        histnorm="probability density",
    ))
    fig.add_trace(go.Scatter(
        x=x_rng, y=norm.pdf(x_rng, mu, std),
        name=f"Normal  μ={mu:.1f}  σ={std:.1f}",
        line=dict(color=COLORS["nvda"], width=2.5),
    ))
    for lo, hi, alpha, nm in [
        (mu - std,   mu + std,   0.15, "±1σ (68%)"),
        (mu - 2*std, mu + 2*std, 0.07, "±2σ (95%)"),
    ]:
        xf = np.linspace(max(0.0, lo), hi, 200)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xf, xf[::-1]]),
            y=np.concatenate([norm.pdf(xf, mu, std), np.zeros(len(xf))]),
            fill="toself", fillcolor=f"rgba(118,185,0,{alpha})",
            line=dict(width=0), name=nm,
        ))
    fig.add_vline(x=curr, line_dash="dash", line_color=COLORS["predict"], line_width=2,
                  annotation_text=f"Current: ${curr:.1f}",
                  annotation_position="top right",
                  annotation_font=dict(color=COLORS["predict"], size=11))
    fig.add_vline(x=mu, line_dash="solid", line_color=COLORS["nvda"], line_width=2,
                  annotation_text=f"Mean: ${mu:.1f}",
                  annotation_position="top left",
                  annotation_font=dict(color=COLORS["nvda"], size=11))
    fig.update_layout(
        **_LAYOUT, height=370, barmode="overlay",
        title=dict(text=f"<b>NVDA Price Distribution — Day +{horizon}</b>",
                   font=dict(size=13)),
        xaxis_title="Price (USD)", yaxis_title="Probability Density",
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def plot_fan_chart(mc: dict, price_df: pd.DataFrame, n_days: int) -> go.Figure:
    last_40   = price_df.tail(40)
    fut_dates = pd.bdate_range(
        start=pd.Timestamp(str(price_df["date"].iloc[-1])) + timedelta(days=1),
        periods=n_days,
    )
    bands = mc["bands"]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_40["date"], y=last_40["close"],
        name="Historical Close", line=dict(color=COLORS["nvda"], width=2),
    ))
    # 5–95 % shaded band
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(bands["pct_95"]) + list(bands["pct_5"][::-1]),
        fill="toself", fillcolor="rgba(69,183,209,0.10)",
        line=dict(width=0), name="5–95th %ile",
    ))
    # 25–75 % shaded band
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(bands["pct_75"]) + list(bands["pct_25"][::-1]),
        fill="toself", fillcolor="rgba(69,183,209,0.20)",
        line=dict(width=0), name="25–75th %ile",
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=bands["pct_50"],
        name="Median forecast",
        line=dict(color=COLORS["predict"], width=2.5, dash="dot"),
    ))
    for band, col, nm in [("pct_5",  COLORS["ci_lo"], "5th %ile"),
                           ("pct_95", COLORS["ci_hi"], "95th %ile")]:
        fig.add_trace(go.Scatter(
            x=fut_dates, y=bands[band],
            name=nm, line=dict(color=col, width=1, dash="dash"),
        ))
    # Connector dot
    fig.add_trace(go.Scatter(
        x=[price_df["date"].iloc[-1], fut_dates[0]],
        y=[float(price_df["close"].iloc[-1]), bands["pct_50"][0]],
        line=dict(color=COLORS["predict"], width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT, height=440,
        title=dict(
            text=f"<b>NVDA {n_days}-Day Price Forecast</b>  ({mc['n_simulations']:,} simulations)",
            font=dict(size=14),
        ),
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=-0.14),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Historical Price Data
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📈 Section 1 — NVDA Historical Price Data")

with st.spinner("Loading NVDA price data from Yahoo Finance…"):
    price_df = fetch_price_data(str(end_date))

if price_df.empty:
    st.error("Could not load NVDA price data. Check the internet connection.")
    st.stop()

curr_close = float(price_df["close"].iloc[-1])
prev_close = float(price_df["close"].iloc[-2])
pct_change = (curr_close - prev_close) / prev_close * 100
yr_high    = float(price_df["close"].tail(252).max())
yr_low     = float(price_df["close"].tail(252).min())
log_rets   = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
ann_vol    = float(log_rets.tail(252).std() * np.sqrt(252) * 100)

k1, k2, k3, k4, k5 = st.columns(5)
_kpi(k1, f"${curr_close:,.2f}", "Current Close",
     COLORS["positive"] if pct_change >= 0 else COLORS["negative"])
_kpi(k2, f"{pct_change:+.2f}%", "Day Change",
     COLORS["positive"] if pct_change >= 0 else COLORS["negative"])
_kpi(k3, f"${yr_high:,.2f}", "52-Week High",  "#F39C12")
_kpi(k4, f"${yr_low:,.2f}",  "52-Week Low",   "#95A5A6")
_kpi(k5, f"{ann_vol:.1f}%",  "Annual Volatility", COLORS["blue"])

st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(plot_candlestick(price_df, str(end_date)), use_container_width=True)

with st.expander("📋 Raw OHLCV — latest 30 rows"):
    st.dataframe(
        price_df.tail(30).sort_values("date", ascending=False)
                .style.format({
                    "open": "${:.2f}", "high": "${:.2f}",
                    "low":  "${:.2f}", "close": "${:.2f}",
                    "volume": "{:,.0f}",
                }),
        use_container_width=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — News Sentiment Analysis
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📰 Section 2 — NVIDIA News Sentiment Analysis")

news_sent  = st.session_state["nvda_news_sent"]
daily_sent = st.session_state["nvda_daily_sent"]

if news_sent is None:
    st.markdown("""
<div class='warn-box'>
  ⚡ Click <b>Fetch Data & Train</b> in the sidebar to pull live NVIDIA headlines,
  run sentiment analysis, train the prediction model, and generate forecasts.
</div>
""", unsafe_allow_html=True)
elif isinstance(news_sent, pd.DataFrame) and news_sent.empty:
    st.warning("No news articles were retrieved. Try increasing the lookback window and re-fetching.")
else:
    total_art = len(news_sent)
    pos_pct   = (news_sent["label"] == "positive").mean() * 100
    neg_pct   = (news_sent["label"] == "negative").mean() * 100
    avg_c     = float(news_sent["compound"].mean())
    cum_c     = float(daily_sent["cumulative_compound"].iloc[-1]) \
                if (daily_sent is not None and not daily_sent.empty) else 0.0

    n1, n2, n3, n4, n5 = st.columns(5)
    _kpi(n1, f"{total_art:,}",    "Total Articles",    COLORS["blue"])
    _kpi(n2, f"{pos_pct:.1f}%",   "📈 Positive",       COLORS["positive"])
    _kpi(n3, f"{neg_pct:.1f}%",   "📉 Negative",       COLORS["negative"])
    _kpi(n4, f"{avg_c:+.3f}",     "Avg Compound",      COLORS["nvda"])
    _kpi(n5, f"{cum_c:+.3f}",
         "Cumulative Score",
         COLORS["positive"] if cum_c > 0 else COLORS["negative"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_article_scatter(news_sent), use_container_width=True)

    # ── Section 3: Cumulative score ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Section 3 — Cumulative News Score")
    if daily_sent is not None and not daily_sent.empty:
        st.plotly_chart(plot_cumulative_sentiment(daily_sent), use_container_width=True)

        pie_col, card_col = st.columns([1, 1.5])
        with pie_col:
            pie_df = news_sent["label"].value_counts().reset_index()
            pie_df.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(
                pie_df, values="Count", names="Sentiment",
                color="Sentiment",
                color_discrete_map={k: COLORS[k] for k in ("positive", "negative", "neutral")},
                hole=0.55,
            )
            fig_pie.update_traces(textinfo="label+percent", textfont_size=12)
            fig_pie.update_layout(
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                font=dict(color="#E8EAED"), height=270, showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="<b>Sentiment Split</b>", font=dict(size=13)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with card_col:
            st.markdown("#### Top Headlines")
            for lbl, emoji in [("positive", "📈"), ("negative", "📉")]:
                subset = news_sent[news_sent["label"] == lbl]
                if lbl == "negative":
                    subset = subset.nsmallest(3, "compound")
                else:
                    subset = subset.nlargest(3, "compound")
                for _, row in subset.iterrows():
                    title_txt = str(row.get("title", ""))[:100]
                    src       = row.get("source", "")
                    url       = row.get("url", "#")
                    c_sc      = float(row.get("compound", 0))
                    pub       = str(row.get("date", ""))[:10]
                    st.markdown(f"""
<div class='news-card'>
  <span class='badge-{lbl}'>{emoji} {lbl.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{pub} · {src}</small><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:600;font-size:.88rem;'>
    {title_txt}
  </a>
  <br><small style='color:#9AA0B4'>compound: {c_sc:+.3f}</small>
</div>""", unsafe_allow_html=True)

    with st.expander(f"📋 All {total_art} NVIDIA articles"):
        disp = news_sent[["date", "title", "source", "label",
                           "compound", "positive", "negative", "neutral"]].copy()
        disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
        disp.columns = ["Date", "Title", "Source", "Sentiment",
                         "Compound", "Pos", "Neg", "Neu"]
        st.dataframe(disp, use_container_width=True, height=320)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Sentiment Model Evaluation
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🔬 Section 4 — Sentiment Model Evaluation")
st.markdown(
    "Evaluate **VADER** (and optionally **FinBERT**) on 45 balanced "
    "Financial Phrase Bank samples (15 pos · 15 neg · 15 neu). "
    "Metrics: Accuracy, Macro-F1, Precision, Recall, Confusion Matrix."
)

if st.session_state["run_eval"]:
    from src.nvidia_predictor import evaluate_vader, evaluate_finbert, LABELS as SENT_LABELS

    with st.spinner("Evaluating VADER…"):
        vader_res = evaluate_vader()
    st.session_state["eval_results"] = {"vader": vader_res}

    if use_finbert:
        with st.spinner("Evaluating FinBERT via HuggingFace API…"):
            fb_res = evaluate_finbert(HF_TOKEN)
        if fb_res:
            st.session_state["eval_results"]["finbert"] = fb_res
        else:
            st.warning("FinBERT API unavailable — showing VADER only.")

    st.session_state["run_eval"] = False

eval_results = st.session_state.get("eval_results")
if eval_results:
    from src.nvidia_predictor import LABELS as SENT_LABELS
    n_eval_cols = 2 if ("finbert" in eval_results) else 1
    eval_cols   = st.columns(n_eval_cols)
    for col_idx, (model_key, model_lbl) in enumerate(
        [("vader", "🔵 VADER"), ("finbert", "🟢 FinBERT")]
    ):
        if model_key not in eval_results:
            continue
        res = eval_results[model_key]
        with eval_cols[col_idx]:
            st.markdown(f"#### {model_lbl}")
            e1, e2, e3, e4 = st.columns(4)
            _kpi(e1, f"{res['accuracy']:.1%}",    "Accuracy",    COLORS["blue"])
            _kpi(e2, f"{res['macro_f1']:.3f}",    "Macro F1",    COLORS["nvda"])
            _kpi(e3, f"{res['macro_prec']:.3f}",  "Precision",   "#F39C12")
            _kpi(e4, f"{res['macro_recall']:.3f}","Recall",      "#95A5A6")
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(
                plot_conf_matrix(res["confusion_matrix"], SENT_LABELS, model_lbl.split()[1]),
                use_container_width=True,
            )
            pc_rows = []
            for lbl in SENT_LABELS:
                if lbl in res["per_class"]:
                    pc = res["per_class"][lbl]
                    pc_rows.append({
                        "Class":     lbl.capitalize(),
                        "Precision": f"{pc['precision']:.3f}",
                        "Recall":    f"{pc['recall']:.3f}",
                        "F1":        f"{pc['f1-score']:.3f}",
                        "Support":   int(pc["support"]),
                    })
            st.dataframe(pd.DataFrame(pc_rows), hide_index=True, use_container_width=True)
            if res.get("roc_auc"):
                st.markdown(f"**ROC-AUC (macro-OvR):** `{res['roc_auc']:.4f}`")
else:
    st.info("👆 Click **Evaluate Sentiment Models** in the sidebar to run the evaluation.")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Model Training & Evaluation
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🧠 Section 5 — Linear Regression Model Training & Evaluation")

models  = st.session_state["nvda_models"]
feat_df = st.session_state["nvda_feat_df"]
has_s   = st.session_state["nvda_has_s"]

if models is None:
    st.info("Click **Fetch Data & Train** in the sidebar to build the prediction model.")
else:
    st.markdown(f"""
<div class='section-box'>
<b>Architecture:</b> Ridge Regression (L2, α=1.0) &nbsp;|&nbsp;
<b>Target:</b> next-day log-return → USD price &nbsp;|&nbsp;
<b>Split:</b> 80 % train / 20 % test (chronological) &nbsp;|&nbsp;
<b>CV:</b> 5-fold TimeSeriesSplit<br>
<b>Price features:</b> lag returns (1,2,3,5,10d), RSI-14, MACD, Bollinger Band %, SMA20/50 ratio, ATR, volume ratio, momentum (5/20d)
{"<br><b>Sentiment features:</b> daily compound, MA-3, MA-5, pos ratio, neg ratio" if has_s else ""}
</div>
""", unsafe_allow_html=True)

    tab_price, tab_sent = st.tabs([
        "📊 Price-Only Model",
        "🗞️ Sentiment-Augmented Model" + (" ✅" if has_s else " (no news)"),
    ])

    for tab, mk, mlbl in [
        (tab_price, "price_model",     "Price-Only"),
        (tab_sent,  "sentiment_model", "Sentiment-Augmented"),
    ]:
        with tab:
            if mk not in models:
                st.warning("Model not available — not enough data.")
                continue
            mr = models[mk]
            m  = mr["metrics"]

            st.markdown(f"#### {mlbl} — Metrics")
            r1 = st.columns(4)
            r2 = st.columns(4)
            r3 = st.columns(4)
            _kpi(r1[0], f"${m['train_mae_usd']:.2f}",  "Train MAE ($)",       COLORS["blue"])
            _kpi(r1[1], f"${m['test_mae_usd']:.2f}",   "Test MAE ($)",        COLORS["nvda"])
            _kpi(r1[2], f"${m['train_rmse_usd']:.2f}", "Train RMSE ($)",      "#F39C12")
            _kpi(r1[3], f"${m['test_rmse_usd']:.2f}",  "Test RMSE ($)",       "#E74C3C")
            _kpi(r2[0], f"{m['train_r2']:.4f}",        "Train R²",            COLORS["blue"])
            _kpi(r2[1], f"{m['test_r2']:.4f}",         "Test R²",             COLORS["nvda"])
            _kpi(r2[2], f"{m['train_dir_acc']:.1%}",   "Train Direction Acc", COLORS["positive"])
            _kpi(r2[3], f"{m['test_dir_acc']:.1%}",    "Test Direction Acc",  COLORS["positive"])
            _kpi(r3[0], f"{m['cv_r2_mean']:.4f}",      "CV R² mean",          COLORS["blue"])
            _kpi(r3[1], f"±{m['cv_r2_std']:.4f}",      "CV R² std",           "#95A5A6")
            _kpi(r3[2], f"{m['cv_dir_mean']:.1%}",     "CV Direction Acc",    COLORS["nvda"])
            _kpi(r3[3],
                 f"{mr['train_size']:,} / {mr['test_size']:,}",
                 "Train / Test rows", "#95A5A6")
            st.markdown("<br>", unsafe_allow_html=True)

            st.plotly_chart(plot_model_perf(mr, mlbl), use_container_width=True)

            col_fi, col_res = st.columns([1.2, 1])
            with col_fi:
                st.plotly_chart(
                    plot_feat_importance(mr["feature_importance"],
                                         f"{mlbl} Feature Importance"),
                    use_container_width=True,
                )
            with col_res:
                st.plotly_chart(plot_residuals(mr["residuals"]), use_container_width=True)
                try:
                    n_res = min(500, len(mr["residuals"]))
                    _, p_sw = shapiro(mr["residuals"][:n_res])
                    norm_txt = (
                        '<span style="color:#2ECC71">✓ Approximately normal (p > 0.05)</span>'
                        if p_sw > 0.05 else
                        '<span style="color:#F39C12">⚠ Non-normal residuals (p ≤ 0.05)</span>'
                    )
                except Exception:
                    p_sw, norm_txt = float("nan"), "Shapiro–Wilk test unavailable"
                st.markdown(f"""
<div class='section-box' style='font-size:.85rem;'>
<b>Residual stats</b><br>
Mean: <code>{mr['resid_mean']:.6f}</code> &nbsp; Std: <code>{mr['resid_std']:.6f}</code><br>
Shapiro–Wilk p: <code>{p_sw:.4f}</code><br>{norm_txt}
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Price Prediction & Probability Distribution
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🔮 Section 6 — Price Prediction & Probability Distribution")

mc = st.session_state["nvda_mc"]

if mc is None or models is None:
    st.info("Run the pipeline first (click **Fetch Data & Train**).")
else:
    exp_1d = mc["expected_price_1d"]
    exp_rt = mc["predicted_return_1d"]
    pa_1d  = mc["price_at"].get(1, {})

    st.markdown(f"""
<div class='section-box'>
  <b>Active model:</b>
  <span class='info-tag'>{'Sentiment-Augmented' if has_s else 'Price-Only'}</span>
  &nbsp; <b>Simulations:</b> <code>{mc['n_simulations']:,}</code>
  &nbsp; <b>Horizon:</b> <code>{mc['n_days']} business days</code>
  &nbsp; <b>Current price:</b> <code>${curr_close:,.2f}</code>
  &nbsp; <b>Annual vol:</b> <code>{mc['hist_vol_annual']:.1%}</code>
</div>
""", unsafe_allow_html=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    _kpi(f1, f"${curr_close:,.2f}", "Current Price", COLORS["nvda"])
    _kpi(f2, f"${exp_1d:,.2f}", "Expected Day+1",
         COLORS["positive"] if exp_1d >= curr_close else COLORS["negative"])
    _kpi(f3, f"{exp_rt*100:+.3f}%", "Model Return t+1",
         COLORS["positive"] if exp_rt >= 0 else COLORS["negative"])
    _kpi(f4, f"${pa_1d.get('std', 0):,.2f}",   "Price Std (1d)",     "#F39C12")
    _kpi(f5, f"{pa_1d.get('prob_up', 0.5):.1%}", "P(Price > Current)",
         COLORS["positive"] if pa_1d.get("prob_up", 0.5) >= 0.5 else COLORS["negative"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Fan chart
    st.plotly_chart(plot_fan_chart(mc, price_df, mc["n_days"]), use_container_width=True)

    # Probability distributions per horizon
    st.markdown("### 🔔 Price Distributions by Horizon")
    avail_horizons = sorted(mc["price_at"].keys())
    h_tabs = st.tabs([f"Day +{d}" for d in avail_horizons])
    for htab, day in zip(h_tabs, avail_horizons):
        with htab:
            dcol, tcol = st.columns([1.8, 1])
            with dcol:
                st.plotly_chart(plot_prob_dist(mc, day), use_container_width=True)
            with tcol:
                pa = mc["price_at"][day]
                st.dataframe(
                    pd.DataFrame({
                        "Statistic": [
                            "Current Price", "Mean (Expected)", "Median",
                            "Std Dev", "5th %ile (Bear)", "25th %ile",
                            "75th %ile", "95th %ile (Bull)",
                            "P(Price > Current)", "P(+5% gain)",
                            "Annual Volatility",
                        ],
                        "Value": [
                            f"${curr_close:,.2f}",
                            f"${pa['mean']:,.2f}",   f"${pa['median']:,.2f}",
                            f"${pa['std']:,.2f}",    f"${pa['pct_5']:,.2f}",
                            f"${pa['pct_25']:,.2f}", f"${pa['pct_75']:,.2f}",
                            f"${pa['pct_95']:,.2f}",
                            f"{pa['prob_up']:.1%}",
                            f"{pa['prob_up_5pct']:.1%}",
                            f"{mc['hist_vol_annual']:.1%}",
                        ],
                    }),
                    hide_index=True, use_container_width=True, height=420,
                )

    # Multi-horizon summary
    st.markdown("### 📊 Multi-Horizon Summary")
    rows_h = [
        {
            "Horizon":     f"Day +{d}",
            "Mean ($)":    f"${pa['mean']:,.2f}",
            "Median ($)":  f"${pa['median']:,.2f}",
            "Std ($)":     f"${pa['std']:,.2f}",
            "Bear 5%ile":  f"${pa['pct_5']:,.2f}",
            "Bull 95%ile": f"${pa['pct_95']:,.2f}",
            "P(Up)":       f"{pa['prob_up']:.1%}",
            "P(+5%)":      f"{pa['prob_up_5pct']:.1%}",
        }
        for d, pa in sorted(mc["price_at"].items())
    ]
    st.dataframe(pd.DataFrame(rows_h), hide_index=True, use_container_width=True)

    # Individual paths expander
    with st.expander("🔬 Show 50 individual simulated price paths"):
        fut_d = pd.bdate_range(
            start=pd.Timestamp(str(price_df["date"].iloc[-1])) + timedelta(days=1),
            periods=mc["n_days"],
        )
        fig_paths = go.Figure()
        sample_idx = np.random.default_rng(0).choice(
            mc["n_simulations"], min(50, mc["n_simulations"]), replace=False
        )
        for idx in sample_idx:
            fig_paths.add_trace(go.Scatter(
                x=fut_d, y=mc["paths"][idx],
                mode="lines", line=dict(color="rgba(118,185,0,0.18)", width=1),
                showlegend=False,
            ))
        fig_paths.add_trace(go.Scatter(
            x=fut_d, y=mc["bands"]["pct_50"],
            name="Median", line=dict(color=COLORS["predict"], width=2.5),
        ))
        fig_paths.add_hline(
            y=curr_close, line_dash="dash",
            line_color=COLORS["nvda"], line_width=1.5,
            annotation_text=f"Current: ${curr_close:.2f}",
        )
        fig_paths.update_layout(
            **_LAYOUT, height=370,
            title="<b>50 Simulated NVDA Price Paths (GBM)</b>",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig_paths, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9AA0B4;font-size:.80rem;padding:10px 0;'>
  NVIDIA Price Predictor &nbsp;|&nbsp; Yahoo Finance · NewsAPI.org &nbsp;|&nbsp;
  Ridge Regression · Monte Carlo GBM &nbsp;|&nbsp; VADER · FinBERT (ProsusAI)<br>
  <b style='color:#76B900'>Educational purposes only — not financial advice.</b>
</div>
""", unsafe_allow_html=True)
