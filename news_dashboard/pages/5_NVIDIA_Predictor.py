"""
Page 4 – NVIDIA Stock Price Prediction Dashboard
=================================================
End-to-end pipeline:
  1. Fetch NVDA OHLCV data from Yahoo Finance (2015-01-01 → user end date)
  2. Fetch NVIDIA-related headlines from NewsAPI
  3. Sentiment analysis (VADER / FinBERT) with evaluation on Financial Phrase Bank
  4. Cumulative news sentiment score
  5. Linear Regression (Ridge) — price-only & sentiment-augmented — with CV metrics
  6. Monte Carlo / GBM price simulation → probability distribution
  7. Fan-chart future forecast with mean ± confidence bands
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import warnings, requests, time
warnings.filterwarnings("ignore")

from scipy.stats import norm, shapiro
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_API_KEY, HF_TOKEN

# ── Page Config ───────────────────────────────────────────────────────────────

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
.metric-card h2{font-size:1.9rem;margin:4px 0;}
.metric-card p{font-size:.80rem;color:#9AA0B4;margin:0;}
.section-box{background:#1A1D2E;border-radius:12px;padding:20px 24px;
             border:1px solid #2D3153;margin-bottom:16px;}
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

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🟢 NVIDIA Predictor")
    st.markdown("---")

    end_date = st.date_input(
        "Price data end date",
        value=date.today(),
        max_value=date.today(),
        help="OHLCV data is fetched from 2015-01-01 to this date.",
    )
    pred_days = st.slider("Forecast horizon (days)", 5, 60, 30, 5)
    n_sims    = st.select_slider(
        "Monte Carlo simulations",
        options=[1_000, 5_000, 10_000, 50_000],
        value=10_000,
    )
    st.markdown("---")

    news_lookback = st.slider(
        "News lookback (days)",
        7, 30, 20,
        help="NewsAPI free tier supports ~30 days back.",
    )
    max_articles = st.slider("Max articles per query", 10, 100, 50, 10)
    model_choice = st.radio(
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
    if fetch_btn:
        st.cache_data.clear()
        st.session_state["nvidia_run"] = True

    run_eval_btn = st.button(
        "📊  Evaluate Sentiment Models",
        use_container_width=True,
        help="Runs VADER (instant) and optionally FinBERT (HF API) on 45 financial phrases.",
    )
    if run_eval_btn:
        st.session_state["run_eval"] = True

    st.markdown("---")
    st.caption("Data: Yahoo Finance · NewsAPI.org · FinBERT/VADER")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class='nvda-header'>
  <h1>🟢 NVIDIA (NVDA) Price Prediction Dashboard</h1>
  <p>
    Sentiment-augmented Ridge regression &nbsp;|&nbsp; Monte Carlo simulation &nbsp;|&nbsp;
    News: NewsAPI.org &nbsp;|&nbsp; Price: Yahoo Finance (2015 → {end_date})
  </p>
</div>
""", unsafe_allow_html=True)

# ── Data Fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(end_date_str: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(
            TICKER, start=TRAIN_START, end=end_date_str,
            interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            return pd.DataFrame()

        # yfinance ≥ 0.2 returns MultiIndex columns (field, ticker) and
        # a DatetimeIndex with no name — normalise both.
        raw.index.name = "Date"
        df = raw.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            # Take only the level-0 name (field), lowercase it
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # The date column may appear as 'date' or be the first column
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        needed = [c for c in ["date", "open", "high", "low", "close", "volume"]
                  if c in df.columns]
        df = df[needed].dropna()
        return df.sort_values("date").reset_index(drop=True)
    except ImportError:
        st.error("❌ `yfinance` is not installed. Run: `pip install yfinance` in your venv.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
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
                    "q":        query,
                    "from":     str(news_from),
                    "to":       str(news_to),
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": min(max_per_q, 100),
                    "apiKey":   api_key,
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
                    "title":       art.get("title", ""),
                    "description": art.get("description") or "",
                    "url":         url,
                    "source":      (art.get("source") or {}).get("name", ""),
                    "published_at": art.get("publishedAt", ""),
                })
            time.sleep(0.12)
        except Exception:
            continue
    if not articles:
        return pd.DataFrame()
    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["published_at"].dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=["date", "title"]).sort_values("date").reset_index(drop=True)


# ── Sentiment Analysis ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def run_vader(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    sid = SentimentIntensityAnalyzer()
    df  = news_df.copy()
    rows = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row.get('description','')}"[:512]
        sc   = sid.polarity_scores(text)
        c    = sc["compound"]
        rows.append({
            "compound": round(c, 4),
            "positive": round(sc["pos"], 4),
            "negative": round(sc["neg"], 4),
            "neutral":  round(sc["neu"], 4),
            "label":    "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"),
        })
    return df.assign(**pd.DataFrame(rows))


@st.cache_data(ttl=3600, show_spinner=False)
def run_finbert(news_df: pd.DataFrame, hf_token: str) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    hdrs   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    df     = news_df.copy()
    texts  = [f"{r['title']} {r.get('description','')}"[:512] for _, r in df.iterrows()]
    rows   = []
    for i in range(0, len(texts), 8):
        batch = texts[i: i + 8]
        try:
            resp = requests.post(HF_URL, headers=hdrs,
                                 json={"inputs": batch, "options": {"wait_for_model": True}},
                                 timeout=90)
            if resp.status_code == 200:
                for item in resp.json():
                    items = item if isinstance(item, list) else [item]
                    sc    = {r["label"].lower(): r["score"] for r in items}
                    pos, neg = sc.get("positive", 0), sc.get("negative", 0)
                    c = round(pos - neg, 4)
                    rows.append({
                        "compound": c,
                        "positive": round(pos, 4),
                        "negative": round(neg, 4),
                        "neutral":  round(sc.get("neutral", 0), 4),
                        "label":    "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"),
                    })
                continue
        except Exception:
            pass
        rows.extend([{"compound": 0, "positive": 0, "negative": 0, "neutral": 1, "label": "neutral"}
                      for _ in batch])
        time.sleep(0.2)
    extra = len(df) - len(rows)
    if extra > 0:
        rows.extend([{"compound": 0, "positive": 0, "negative": 0, "neutral": 1, "label": "neutral"}] * extra)
    return df.assign(**pd.DataFrame(rows[:len(df)]))


def daily_sentiment(news_sent: pd.DataFrame) -> pd.DataFrame:
    if news_sent.empty:
        return pd.DataFrame()
    df = news_sent.copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("date")
    out = g.agg(
        avg_compound   = ("compound", "mean"),
        total_articles = ("compound", "count"),
        pos_count      = ("label",    lambda x: (x == "positive").sum()),
        neg_count      = ("label",    lambda x: (x == "negative").sum()),
        neu_count      = ("label",    lambda x: (x == "neutral").sum()),
    ).reset_index().sort_values("date")
    out["cumulative_compound"] = out["avg_compound"].cumsum()
    out["compound_ma3"]        = out["avg_compound"].rolling(3, min_periods=1).mean()
    out["compound_ma5"]        = out["avg_compound"].rolling(5, min_periods=1).mean()
    out["pos_ratio"]           = out["pos_count"] / out["total_articles"]
    out["neg_ratio"]           = out["neg_count"] / out["total_articles"]
    return out


# ── ML Pipeline ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def build_and_train(_price_df: pd.DataFrame, _daily_sent: pd.DataFrame) -> dict:
    """Wrapper so Streamlit can cache the result; leading _ suppresses hashing."""
    from src.nvidia_predictor import (
        compute_technical_features, merge_sentiment,
        train_price_models, SENTIMENT_FEATURES,
    )
    feat_df = compute_technical_features(_price_df)
    has_s   = not _daily_sent.empty
    if has_s:
        feat_df = merge_sentiment(feat_df, _daily_sent)
    else:
        for c in SENTIMENT_FEATURES:
            feat_df[c] = 0.0
    models = train_price_models(feat_df, has_sentiment=has_s)
    return {"feat_df": feat_df, "models": models, "has_sentiment": has_s}


@st.cache_data(ttl=1800, show_spinner=False)
def run_monte_carlo(_model_result: dict, _price_df: pd.DataFrame,
                    _feat_df: pd.DataFrame, n_days: int, n_sims: int) -> dict:
    from src.nvidia_predictor import monte_carlo_forecast
    return monte_carlo_forecast(
        _model_result, _price_df, _feat_df,
        n_days=n_days, n_simulations=n_sims,
    )


# ── Plotting Helpers ──────────────────────────────────────────────────────────

def plot_candlestick(price_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=price_df["date"], open=price_df["open"], high=price_df["high"],
        low=price_df["low"],  close=price_df["close"],
        increasing_line_color=COLORS["nvda"],
        decreasing_line_color=COLORS["negative"],
        name="NVDA", showlegend=False,
    ), row=1, col=1)
    # SMA overlays
    for w, col in [(50, "#F39C12"), (200, "#4ECDC4")]:
        sma = price_df["close"].rolling(w).mean()
        fig.add_trace(go.Scatter(
            x=price_df["date"], y=sma, name=f"SMA{w}",
            line=dict(color=col, width=1.2, dash="dot"),
        ), row=1, col=1)
    # Volume bars
    colours = [COLORS["nvda"] if c >= o else COLORS["negative"]
               for c, o in zip(price_df["close"], price_df["open"])]
    fig.add_trace(go.Bar(
        x=price_df["date"], y=price_df["volume"],
        name="Volume", marker_color=colours, opacity=0.6, showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        **_LAYOUT,
        height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        title=dict(text=f"<b>NVDA</b> Historical Price (2015 → {end_date})",
                   font=dict(size=14, color="#E8EAED")),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def plot_cumulative_sentiment(daily_s: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.04,
                        subplot_titles=["Cumulative News Score", "Daily Avg Compound Score"])
    # Cumulative
    fig.add_trace(go.Scatter(
        x=daily_s["date"], y=daily_s["cumulative_compound"],
        fill="tozeroy", fillcolor="rgba(118,185,0,0.15)",
        line=dict(color=COLORS["nvda"], width=2),
        name="Cumulative Score",
    ), row=1, col=1)
    # Daily bars
    bar_cols = [
        COLORS["positive"] if v > 0.05 else (COLORS["negative"] if v < -0.05 else COLORS["neutral"])
        for v in daily_s["avg_compound"]
    ]
    fig.add_trace(go.Bar(
        x=daily_s["date"], y=daily_s["avg_compound"],
        marker_color=bar_cols, opacity=0.7, name="Daily Compound",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=daily_s["date"], y=daily_s["compound_ma5"],
        line=dict(color=COLORS["predict"], width=2), name="5-Day MA",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1, row=2, col=1)
    fig.update_layout(**_LAYOUT, height=420,
                      legend=dict(orientation="h", y=-0.08))
    return fig


def plot_model_performance(model_res: dict, label: str = "Test Set") -> go.Figure:
    dates  = pd.to_datetime(model_res["test_dates"])
    actual = model_res["test_price_actual"]
    pred   = model_res["test_price_predicted"]
    close  = model_res["test_close"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual, name="Actual Price",
        line=dict(color=COLORS["nvda"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred, name="Predicted Price",
        line=dict(color=COLORS["predict"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **_LAYOUT, height=350,
        title=dict(text=f"<b>Actual vs Predicted NVDA Price</b> — {label}",
                   font=dict(size=13)),
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_residuals(residuals: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residual Distribution", "Residuals Over Time"])
    # Histogram + normal fit
    mu, std = float(np.mean(residuals)), float(np.std(residuals))
    x_range = np.linspace(mu - 4*std, mu + 4*std, 200)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, name="Residuals",
        marker_color=COLORS["blue"], opacity=0.65,
        histnorm="probability density",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_range, y=norm.pdf(x_range, mu, std),
        name="Normal fit", line=dict(color=COLORS["predict"], width=2),
    ), row=1, col=1)
    # Scatter over time
    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))), y=residuals,
        mode="markers", name="Residual",
        marker=dict(color=COLORS["blue"], size=4, opacity=0.6),
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1, row=1, col=2)
    fig.update_layout(**_LAYOUT, height=320,
                      showlegend=False)
    return fig


def plot_feature_importance(feat_imp: pd.Series, title: str) -> go.Figure:
    from src.nvidia_predictor import FEATURE_LABELS
    top = feat_imp.head(12)
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
        **_LAYOUT, height=380,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13)),
        xaxis_title="Absolute Coefficient (scaled)",
        margin=dict(l=120, r=20, t=50, b=20),
    )
    return fig


def plot_confusion_matrix(cm: list, labels: list, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, CARD_BG], [0.5, "#1B4B8A"], [1, COLORS["nvda"]]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=14, color="#E8EAED"),
        showscale=False,
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT, height=280,
        title=dict(text=f"<b>{title} Confusion Matrix</b>", font=dict(size=13)),
        xaxis=dict(title="Predicted", side="bottom", gridcolor="transparent"),
        yaxis=dict(title="Actual", gridcolor="transparent"),
        margin=dict(l=80, r=20, t=50, b=60),
    )
    return fig


def plot_probability_distribution(mc: dict, horizon: int = 1) -> go.Figure:
    """Bell curve + histogram for price at a given horizon."""
    pa   = mc["price_at"].get(horizon, mc["price_at"][min(mc["price_at"].keys())])
    curr = mc["current_price"]
    mu, std = pa["mean"], pa["std"]

    # Monte Carlo histogram
    sim_prices = mc["paths"][:, horizon - 1]
    x_range = np.linspace(max(0, mu - 4*std), mu + 4*std, 400)

    fig = go.Figure()
    # Histogram of simulated prices
    fig.add_trace(go.Histogram(
        x=sim_prices, nbinsx=80, name="Simulated Prices",
        marker_color=COLORS["blue"], opacity=0.5,
        histnorm="probability density",
    ))
    # Normal distribution overlay
    fig.add_trace(go.Scatter(
        x=x_range, y=norm.pdf(x_range, mu, std),
        name=f"Normal (μ={mu:.1f}, σ={std:.1f})",
        line=dict(color=COLORS["nvda"], width=2.5),
    ))
    # ±1σ / ±2σ shading
    for lo, hi, alpha, label in [
        (mu - std,   mu + std,   0.15, "±1σ (68%)"),
        (mu - 2*std, mu + 2*std, 0.07, "±2σ (95%)"),
    ]:
        x_fill = np.linspace(max(0, lo), hi, 200)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([norm.pdf(x_fill, mu, std),
                              np.zeros(len(x_fill))]),
            fill="toself", fillcolor=f"rgba(118,185,0,{alpha})",
            line=dict(width=0), name=label, showlegend=True,
        ))
    # Current price line
    fig.add_vline(x=curr, line_dash="dash", line_color=COLORS["predict"], line_width=2,
                  annotation_text=f"Current: ${curr:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color=COLORS["predict"], size=11))
    # Mean line
    fig.add_vline(x=mu, line_dash="solid", line_color=COLORS["nvda"], line_width=2,
                  annotation_text=f"Mean: ${mu:.2f}",
                  annotation_position="top left",
                  annotation_font=dict(color=COLORS["nvda"], size=11))
    fig.update_layout(
        **_LAYOUT, height=380,
        title=dict(text=f"<b>NVDA Price Distribution — Day +{horizon}</b>",
                   font=dict(size=13)),
        xaxis_title="Price (USD)",
        yaxis_title="Probability Density",
        legend=dict(orientation="h", y=-0.18),
        barmode="overlay",
    )
    return fig


def plot_fan_chart(mc: dict, price_df: pd.DataFrame, n_days: int) -> go.Figure:
    """Forecast fan chart with historical tail + future bands."""
    last_40 = price_df.tail(40).copy()
    fut_dates = pd.bdate_range(
        start=pd.Timestamp(price_df["date"].iloc[-1]) + timedelta(days=1),
        periods=n_days,
    )
    bands = mc["bands"]
    fig = go.Figure()

    # Historical close
    fig.add_trace(go.Scatter(
        x=last_40["date"], y=last_40["close"],
        name="Historical Close",
        line=dict(color=COLORS["nvda"], width=2),
    ))
    # 5–95 % band
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(bands["pct_95"]) + list(bands["pct_5"][::-1]),
        fill="toself", fillcolor="rgba(69,183,209,0.12)",
        line=dict(width=0), name="5th–95th %ile",
    ))
    # 25–75 % band
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(bands["pct_75"]) + list(bands["pct_25"][::-1]),
        fill="toself", fillcolor="rgba(69,183,209,0.22)",
        line=dict(width=0), name="25th–75th %ile",
    ))
    # Median path
    fig.add_trace(go.Scatter(
        x=fut_dates, y=bands["pct_50"],
        name="Median Forecast",
        line=dict(color=COLORS["predict"], width=2.5, dash="dot"),
    ))
    # 5th / 95th borders
    for band, col, nm in [("pct_5", COLORS["ci_lo"], "5th %ile"),
                           ("pct_95", COLORS["ci_hi"], "95th %ile")]:
        fig.add_trace(go.Scatter(
            x=fut_dates, y=bands[band],
            name=nm, line=dict(color=col, width=1, dash="dash"),
        ))
    # Connector from last close to first forecast
    fig.add_trace(go.Scatter(
        x=[price_df["date"].iloc[-1], fut_dates[0]],
        y=[price_df["close"].iloc[-1], bands["pct_50"][0]],
        line=dict(color=COLORS["predict"], width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT, height=450,
        title=dict(text=f"<b>NVDA {n_days}-Day Price Forecast</b> ({mc['n_simulations']:,} simulations)",
                   font=dict(size=14)),
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=-0.14),
    )
    return fig


def plot_sentiment_article_scatter(news_s: pd.DataFrame) -> go.Figure:
    df = news_s.copy()
    df["short_title"] = df["title"].str[:60] + "…"
    color_map = {
        "positive": COLORS["positive"],
        "negative": COLORS["negative"],
        "neutral":  COLORS["neutral"],
    }
    fig = px.scatter(
        df, x="date", y="compound",
        color="label",
        color_discrete_map=color_map,
        hover_data={"short_title": True, "source": True, "compound": ":.3f"},
        labels={"compound": "Compound Score", "date": "Date", "label": "Sentiment"},
    )
    fig.add_hline(y=0.05,  line_dash="dot", line_color=COLORS["positive"],
                  annotation_text="+0.05 (positive threshold)", line_width=1)
    fig.add_hline(y=-0.05, line_dash="dot", line_color=COLORS["negative"],
                  annotation_text="-0.05 (negative threshold)", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9AA0B4", line_width=1)
    fig.update_layout(
        **_LAYOUT, height=320,
        title=dict(text="<b>Article-Level Sentiment Scores</b>",
                   font=dict(size=13)),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main Logic ────────────────────────────────────────────────────────────────

# ── STEP 1: Price Data ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📈 Section 1 — NVDA Historical Price Data")

with st.spinner("Loading NVDA price data from Yahoo Finance…"):
    price_df = fetch_price_data(str(end_date))

if price_df.empty:
    st.error("Could not load NVDA price data. Check your internet connection or install yfinance.")
    st.stop()

# KPIs
curr_close = float(price_df["close"].iloc[-1])
prev_close = float(price_df["close"].iloc[-2])
pct_change = (curr_close - prev_close) / prev_close * 100
yr_high    = float(price_df["close"].tail(252).max())
yr_low     = float(price_df["close"].tail(252).min())
log_rets   = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
ann_vol    = float(log_rets.tail(252).std() * np.sqrt(252) * 100)

c1, c2, c3, c4, c5 = st.columns(5)
def _kpi(col, val, lbl, color="#4ECDC4"):
    col.markdown(
        f"<div class='metric-card'><h2 style='color:{color}'>{val}</h2><p>{lbl}</p></div>",
        unsafe_allow_html=True,
    )

_kpi(c1, f"${curr_close:,.2f}", "Current Close",
     COLORS["positive"] if pct_change >= 0 else COLORS["negative"])
_kpi(c2, f"{pct_change:+.2f}%",  "Day Change",
     COLORS["positive"] if pct_change >= 0 else COLORS["negative"])
_kpi(c3, f"${yr_high:,.2f}", "52-Week High",  "#F39C12")
_kpi(c4, f"${yr_low:,.2f}",  "52-Week Low",   "#95A5A6")
_kpi(c5, f"{ann_vol:.1f}%",  "Annual Volatility", COLORS["blue"])

st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(plot_candlestick(price_df), use_container_width=True)

with st.expander("📋 View Raw OHLCV Data (latest 30 rows)"):
    st.dataframe(
        price_df.tail(30).sort_values("date", ascending=False)
                .style.format({"open":"${:.2f}","high":"${:.2f}","low":"${:.2f}",
                                "close":"${:.2f}","volume":"{:,.0f}"}),
        use_container_width=True,
    )

# ── STEP 2: News Fetch & Sentiment ───────────────────────────────────────────
st.markdown("---")
st.markdown("## 📰 Section 2 — NVIDIA News Sentiment Analysis")

if not st.session_state.get("nvidia_run") and "news_df" not in st.session_state:
    st.markdown("""
<div class='warn-box'>
  ⚡ Click <b>Fetch Data & Train</b> in the sidebar to pull live NVIDIA news,
  run sentiment analysis, train models, and generate price forecasts.
</div>
""", unsafe_allow_html=True)
else:
    # ── Fetch news ────────────────────────────────────────────────────────
    with st.spinner(f"Fetching NVIDIA news (last {news_lookback} days)…"):
        news_df = fetch_nvidia_news(NEWS_API_KEY, news_lookback, max_articles)

    if news_df.empty:
        st.warning("No news articles returned. Check NewsAPI key or try increasing lookback.")
        news_sent = pd.DataFrame()
        daily_sent = pd.DataFrame()
    else:
        # ── Sentiment ─────────────────────────────────────────────────────
        model_lbl = "FinBERT" if use_finbert else "VADER"
        with st.spinner(f"Running {model_lbl} sentiment on {len(news_df)} articles…"):
            if use_finbert:
                news_sent = run_finbert(news_df, HF_TOKEN)
            else:
                news_sent = run_vader(news_df)
        daily_sent = daily_sentiment(news_sent)

        # ── Metrics ───────────────────────────────────────────────────────
        total_art = len(news_sent)
        pos_pct = (news_sent["label"] == "positive").mean() * 100
        neg_pct = (news_sent["label"] == "negative").mean() * 100
        neu_pct = (news_sent["label"] == "neutral").mean()  * 100
        avg_c   = float(news_sent["compound"].mean())
        cum_c   = float(daily_sent["cumulative_compound"].iloc[-1]) if not daily_sent.empty else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        _kpi(k1, f"{total_art:,}",       "Total Articles",         COLORS["blue"])
        _kpi(k2, f"{pos_pct:.1f}%",      "📈 Positive",            COLORS["positive"])
        _kpi(k3, f"{neg_pct:.1f}%",      "📉 Negative",            COLORS["negative"])
        _kpi(k4, f"{avg_c:+.3f}",        "Avg Compound Score",     COLORS["nvda"])
        _kpi(k5, f"{cum_c:+.3f}",        "Cumulative Score",
             COLORS["positive"] if cum_c > 0 else COLORS["negative"])
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Scatter plot ─────────────────────────────────────────────────
        st.plotly_chart(plot_sentiment_article_scatter(news_sent), use_container_width=True)

        # ── Cumulative score ─────────────────────────────────────────────
        st.markdown("### 📊 Section 3 — Cumulative News Score")
        if not daily_sent.empty:
            st.plotly_chart(plot_cumulative_sentiment(daily_sent, price_df),
                            use_container_width=True)

            # Pie chart of sentiment split
            pie_col, stat_col = st.columns([1, 1.5])
            with pie_col:
                pie_data = news_sent["label"].value_counts().reset_index()
                pie_data.columns = ["Sentiment", "Count"]
                fig_pie = px.pie(pie_data, values="Count", names="Sentiment",
                                 color="Sentiment",
                                 color_discrete_map={
                                     "positive": COLORS["positive"],
                                     "negative": COLORS["negative"],
                                     "neutral":  COLORS["neutral"],
                                 }, hole=0.55)
                fig_pie.update_traces(textinfo="label+percent", textfont_size=12)
                fig_pie.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                                      font=dict(color="#E8EAED"), height=280,
                                      showlegend=False, margin=dict(l=10,r=10,t=20,b=10),
                                      title=dict(text="<b>Sentiment Distribution</b>",
                                                 font=dict(size=13)))
                st.plotly_chart(fig_pie, use_container_width=True)

            with stat_col:
                st.markdown("#### Top Headlines by Sentiment")
                for label_filter, emoji in [("positive","📈"), ("negative","📉")]:
                    subset = news_sent[news_sent["label"] == label_filter].nlargest(
                        3, "compound" if label_filter == "positive" else "negative"
                    )
                    for _, row in subset.iterrows():
                        badge_cls = f"badge-{label_filter}"
                        title_txt = str(row.get("title",""))[:100]
                        src  = row.get("source","")
                        url  = row.get("url","#")
                        c_sc = float(row.get("compound",0))
                        pub  = str(row.get("date",""))[:10]
                        st.markdown(f"""
<div class='news-card'>
  <span class='{badge_cls}'>{emoji} {label_filter.capitalize()}</span>
  &nbsp;<small style='color:#9AA0B4'>{pub} · {src}</small><br>
  <a href='{url}' target='_blank'
     style='color:#E8EAED;text-decoration:none;font-weight:600;font-size:.88rem;'>
    {title_txt}
  </a>
  <br><small style='color:#9AA0B4'>compound: {c_sc:+.3f}</small>
</div>""", unsafe_allow_html=True)

        # ── All Headlines Table ───────────────────────────────────────────
        with st.expander(f"📋 View all {total_art} NVIDIA articles"):
            disp = news_sent[["date","title","source","label","compound",
                               "positive","negative","neutral"]].copy()
            disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
            disp.columns = ["Date","Title","Source","Sentiment",
                             "Compound","Pos","Neg","Neu"]
            st.dataframe(disp, use_container_width=True, height=320)

# ── STEP 3: Sentiment Model Evaluation ───────────────────────────────────────
st.markdown("---")
st.markdown("## 🔬 Section 4 — Sentiment Model Evaluation")
st.markdown("""
Evaluate **VADER** (and optionally **FinBERT**) on 45 balanced Financial Phrase Bank samples
(15 positive · 15 negative · 15 neutral). Metrics: Accuracy, Macro-F1, Precision, Recall,
Confusion Matrix.
""")

if st.session_state.get("run_eval"):
    from src.nvidia_predictor import evaluate_vader, evaluate_finbert, LABELS as SENT_LABELS

    with st.spinner("Running VADER evaluation on Financial Phrase Bank…"):
        vader_eval = evaluate_vader()

    eval_cols = st.columns(2 if use_finbert else 1)

    with eval_cols[0]:
        st.markdown("#### 🔵 VADER Results")
        m1, m2, m3, m4 = st.columns(4)
        _kpi(m1, f"{vader_eval['accuracy']:.1%}",    "Accuracy",       COLORS["blue"])
        _kpi(m2, f"{vader_eval['macro_f1']:.3f}",    "Macro F1",       COLORS["nvda"])
        _kpi(m3, f"{vader_eval['macro_prec']:.3f}",  "Macro Precision","#F39C12")
        _kpi(m4, f"{vader_eval['macro_recall']:.3f}","Macro Recall",   "#95A5A6")
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(plot_confusion_matrix(
            vader_eval["confusion_matrix"], SENT_LABELS, "VADER"
        ), use_container_width=True)
        # Per-class table
        pc_data = []
        for lbl in SENT_LABELS:
            if lbl in vader_eval["per_class"]:
                pc = vader_eval["per_class"][lbl]
                pc_data.append({
                    "Class": lbl.capitalize(),
                    "Precision": f"{pc['precision']:.3f}",
                    "Recall":    f"{pc['recall']:.3f}",
                    "F1":        f"{pc['f1-score']:.3f}",
                    "Support":   int(pc["support"]),
                })
        st.dataframe(pd.DataFrame(pc_data), hide_index=True, use_container_width=True)

    if use_finbert and len(eval_cols) > 1:
        with eval_cols[1]:
            st.markdown("#### 🟢 FinBERT Results")
            with st.spinner("Running FinBERT evaluation via HuggingFace API…"):
                finbert_eval = evaluate_finbert(HF_TOKEN)
            if finbert_eval:
                m1, m2, m3, m4 = st.columns(4)
                _kpi(m1, f"{finbert_eval['accuracy']:.1%}",    "Accuracy",       COLORS["nvda"])
                _kpi(m2, f"{finbert_eval['macro_f1']:.3f}",    "Macro F1",       COLORS["blue"])
                _kpi(m3, f"{finbert_eval['macro_prec']:.3f}",  "Macro Precision","#F39C12")
                _kpi(m4, f"{finbert_eval['macro_recall']:.3f}","Macro Recall",   "#95A5A6")
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(plot_confusion_matrix(
                    finbert_eval["confusion_matrix"], SENT_LABELS, "FinBERT"
                ), use_container_width=True)
                pc_data = []
                for lbl in SENT_LABELS:
                    if lbl in finbert_eval["per_class"]:
                        pc = finbert_eval["per_class"][lbl]
                        pc_data.append({
                            "Class": lbl.capitalize(),
                            "Precision": f"{pc['precision']:.3f}",
                            "Recall":    f"{pc['recall']:.3f}",
                            "F1":        f"{pc['f1-score']:.3f}",
                            "Support":   int(pc["support"]),
                        })
                st.dataframe(pd.DataFrame(pc_data), hide_index=True, use_container_width=True)
                if finbert_eval.get("roc_auc"):
                    st.markdown(f"**ROC-AUC (macro-OvR):** `{finbert_eval['roc_auc']:.4f}`")
            else:
                st.warning("FinBERT API unavailable. Check HF_TOKEN in Streamlit secrets.")

    st.session_state["run_eval"] = False
else:
    st.info("👆 Click **Evaluate Sentiment Models** in the sidebar to run model evaluation.")

# ── STEP 4: Model Training ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🧠 Section 5 — Linear Regression Model Training & Evaluation")

if not st.session_state.get("nvidia_run") and "train_results" not in st.session_state:
    st.info("Click **Fetch Data & Train** in the sidebar to build the prediction model.")
else:
    # Ensure daily_sent exists
    if "daily_sent" not in dir():
        daily_sent = pd.DataFrame()
    if "news_sent" not in dir():
        news_sent = pd.DataFrame()

    with st.spinner("Engineering features and training Ridge regression models…"):
        train_out = build_and_train(price_df, daily_sent)

    feat_df   = train_out["feat_df"]
    models    = train_out["models"]
    has_s     = train_out["has_sentiment"]

    if not models:
        st.error("Model training failed — not enough data.")
    else:
        st.markdown("""
<div class='section-box'>
<b>Model Architecture</b>: Ridge Regression (L2-regularised Linear Regression, α=1.0)<br>
<b>Target</b>: Next-day log-return → converted to USD price for evaluation<br>
<b>Split</b>: 80% train / 20% test (chronological, no look-ahead)<br>
<b>CV</b>: 5-fold TimeSeriesSplit with rolling expanding window<br>
<b>Features</b>: Lag returns, RSI-14, MACD, Bollinger Band %, SMA ratios, ATR, Volume, Momentum
{sent_note}
</div>
""".format(sent_note="+ News sentiment scores (daily compound, MA-3, MA-5, pos/neg ratio)"
           if has_s else ""), unsafe_allow_html=True)

        tab_price, tab_sent = st.tabs([
            "📊 Price-Only Model",
            "🗞️ Sentiment-Augmented Model" + (" (Active)" if has_s else " (No news data)"),
        ])

        for tab, model_key, tab_title in [
            (tab_price, "price_model",     "Price-Only"),
            (tab_sent,  "sentiment_model", "Sentiment-Augmented"),
        ]:
            with tab:
                if model_key not in models:
                    st.warning("Model not available.")
                    continue
                mr = models[model_key]
                m  = mr["metrics"]

                # ── Metrics grid ─────────────────────────────────────────
                st.markdown(f"#### {tab_title} Model — Metrics")
                st.markdown("""
<style>
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

                r1 = st.columns(4)
                r2 = st.columns(4)
                r3 = st.columns(4)

                _kpi(r1[0], f"${m['train_mae_usd']:.2f}",  "Train MAE ($)",      COLORS["blue"])
                _kpi(r1[1], f"${m['test_mae_usd']:.2f}",   "Test MAE ($)",       COLORS["nvda"])
                _kpi(r1[2], f"${m['train_rmse_usd']:.2f}", "Train RMSE ($)",     "#F39C12")
                _kpi(r1[3], f"${m['test_rmse_usd']:.2f}",  "Test RMSE ($)",      "#E74C3C")

                _kpi(r2[0], f"{m['train_r2']:.4f}",        "Train R²",           COLORS["blue"])
                _kpi(r2[1], f"{m['test_r2']:.4f}",         "Test R²",            COLORS["nvda"])
                _kpi(r2[2], f"{m['train_dir_acc']:.1%}",   "Train Direction Acc",COLORS["positive"])
                _kpi(r2[3], f"{m['test_dir_acc']:.1%}",    "Test Direction Acc", COLORS["positive"])

                _kpi(r3[0], f"{m['cv_r2_mean']:.4f}",      "CV R² (mean)",       COLORS["blue"])
                _kpi(r3[1], f"±{m['cv_r2_std']:.4f}",      "CV R² (std)",        "#95A5A6")
                _kpi(r3[2], f"{m['cv_dir_mean']:.1%}",     "CV Direction Acc",   COLORS["nvda"])
                _kpi(r3[3], f"{mr['train_size']:,} / {mr['test_size']:,}",
                     "Train / Test rows", "#95A5A6")
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Actual vs Predicted chart ─────────────────────────────
                st.plotly_chart(plot_model_performance(mr, tab_title), use_container_width=True)

                # ── Feature importance + Residuals side by side ───────────
                col_fi, col_res = st.columns([1.2, 1])
                with col_fi:
                    st.plotly_chart(
                        plot_feature_importance(mr["feature_importance"],
                                                f"{tab_title} Feature Importance"),
                        use_container_width=True,
                    )
                with col_res:
                    st.plotly_chart(plot_residuals(mr["residuals"]),
                                    use_container_width=True)
                    # Shapiro-Wilk normality test
                    stat_sw, p_sw = shapiro(mr["residuals"][:min(500, len(mr["residuals"]))])
                    st.markdown(f"""
<div class='section-box'>
<b>Residual Statistics</b><br>
Mean: <code>{mr['resid_mean']:.6f}</code> &nbsp; Std: <code>{mr['resid_std']:.6f}</code><br>
Shapiro–Wilk p-value: <code>{p_sw:.4f}</code>
{'<span style="color:#2ECC71">✓ Residuals are approximately normal (p > 0.05)</span>'
 if p_sw > 0.05 else
 '<span style="color:#F39C12">⚠ Residuals deviate from normality (p ≤ 0.05)</span>'}
</div>""", unsafe_allow_html=True)

    # ── STEP 5: Price Prediction ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔮 Section 6 — NVDA Price Prediction & Probability Distribution")

    if models:
        # Use sentiment model if news data is available, otherwise price model
        active_key = "sentiment_model" if (has_s and "sentiment_model" in models) else "price_model"
        active_mr  = models[active_key]

        st.markdown(f"""
<div class='section-box'>
  <b>Active model:</b> <span class='info-tag'>{active_key.replace('_',' ').title()}</span>
  &nbsp; <b>Simulations:</b> <code>{n_sims:,}</code>
  &nbsp; <b>Horizon:</b> <code>{pred_days} business days</code>
  &nbsp; <b>Current price:</b> <code>${curr_close:,.2f}</code>
</div>
""", unsafe_allow_html=True)

        with st.spinner(f"Running Monte Carlo ({n_sims:,} simulations × {pred_days} days)…"):
            mc = run_monte_carlo(active_mr, price_df, feat_df, pred_days, n_sims)

        if mc:
            exp_1d = mc["expected_price_1d"]
            exp_ret= mc["predicted_return_1d"]
            pa_1d  = mc["price_at"].get(1, {})

            # ── KPI row ───────────────────────────────────────────────────
            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            _kpi(kc1, f"${curr_close:,.2f}",
                 "Current Price", COLORS["nvda"])
            _kpi(kc2, f"${exp_1d:,.2f}",
                 "Expected Next-Day",
                 COLORS["positive"] if exp_1d >= curr_close else COLORS["negative"])
            _kpi(kc3, f"{exp_ret*100:+.3f}%",
                 "Model Return t+1",
                 COLORS["positive"] if exp_ret >= 0 else COLORS["negative"])
            _kpi(kc4, f"${pa_1d.get('std', 0):,.2f}",
                 "Price Std Dev (1d)", "#F39C12")
            _kpi(kc5, f"{pa_1d.get('prob_up', 0.5):.1%}",
                 "P(Price > Current)",
                 COLORS["positive"] if pa_1d.get("prob_up", 0.5) >= 0.5 else COLORS["negative"])
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Fan chart ─────────────────────────────────────────────────
            st.plotly_chart(plot_fan_chart(mc, price_df, pred_days), use_container_width=True)

            # ── Probability Distributions ─────────────────────────────────
            st.markdown("### 🔔 Price Probability Distributions")
            avail_horizons = sorted(mc["price_at"].keys())
            h_tabs = st.tabs([f"Day +{d}" for d in avail_horizons])
            for htab, day in zip(h_tabs, avail_horizons):
                with htab:
                    col_dist, col_tbl = st.columns([1.8, 1])
                    with col_dist:
                        st.plotly_chart(
                            plot_probability_distribution(mc, day),
                            use_container_width=True,
                        )
                    with col_tbl:
                        pa = mc["price_at"][day]
                        tbl_data = {
                            "Statistic": [
                                "Current Price",
                                "Mean (Expected)",
                                "Median",
                                "Std Dev",
                                "5th Percentile (Bear)",
                                "25th Percentile",
                                "75th Percentile",
                                "95th Percentile (Bull)",
                                "P(Price > Current)",
                                "P(Price > +5%)",
                                "Annual Vol",
                            ],
                            "Value": [
                                f"${curr_close:,.2f}",
                                f"${pa['mean']:,.2f}",
                                f"${pa['median']:,.2f}",
                                f"${pa['std']:,.2f}",
                                f"${pa['pct_5']:,.2f}",
                                f"${pa['pct_25']:,.2f}",
                                f"${pa['pct_75']:,.2f}",
                                f"${pa['pct_95']:,.2f}",
                                f"{pa['prob_up']:.1%}",
                                f"{pa['prob_up_5pct']:.1%}",
                                f"{mc['hist_vol_annual']:.1%}",
                            ],
                        }
                        st.dataframe(
                            pd.DataFrame(tbl_data),
                            hide_index=True,
                            use_container_width=True,
                            height=420,
                        )

            # ── Horizon summary table ─────────────────────────────────────
            st.markdown("### 📊 Multi-Horizon Price Summary")
            horizon_rows = []
            for d, pa in sorted(mc["price_at"].items()):
                horizon_rows.append({
                    "Horizon":    f"Day +{d}",
                    "Mean ($)":   f"${pa['mean']:,.2f}",
                    "Median ($)": f"${pa['median']:,.2f}",
                    "Std ($)":    f"${pa['std']:,.2f}",
                    "Bear (5th %)": f"${pa['pct_5']:,.2f}",
                    "Bull (95th %)": f"${pa['pct_95']:,.2f}",
                    "P(Up)":      f"{pa['prob_up']:.1%}",
                    "P(+5%)":     f"{pa['prob_up_5pct']:.1%}",
                })
            st.dataframe(pd.DataFrame(horizon_rows), hide_index=True,
                         use_container_width=True)

            # ── Simulated Paths Fan ───────────────────────────────────────
            with st.expander("🔬 Show 50 Individual Simulated Price Paths"):
                fut_dates_sample = pd.bdate_range(
                    start=pd.Timestamp(price_df["date"].iloc[-1]) + timedelta(days=1),
                    periods=pred_days,
                )
                fig_paths = go.Figure()
                sample_idx = np.random.choice(n_sims, min(50, n_sims), replace=False)
                for i, idx in enumerate(sample_idx):
                    fig_paths.add_trace(go.Scatter(
                        x=fut_dates_sample,
                        y=mc["paths"][idx],
                        mode="lines",
                        line=dict(color="rgba(118,185,0,0.18)", width=1),
                        showlegend=False,
                    ))
                # Add median
                fig_paths.add_trace(go.Scatter(
                    x=fut_dates_sample, y=mc["bands"]["pct_50"],
                    name="Median", line=dict(color=COLORS["predict"], width=2.5),
                ))
                fig_paths.add_hline(y=curr_close, line_dash="dash",
                                    line_color=COLORS["nvda"], line_width=1.5,
                                    annotation_text=f"Current: ${curr_close:.2f}")
                fig_paths.update_layout(
                    **_LAYOUT, height=380,
                    title="<b>50 Simulated NVDA Price Paths (GBM)</b>",
                    yaxis_title="Price (USD)",
                )
                st.plotly_chart(fig_paths, use_container_width=True)
        else:
            st.error("Monte Carlo simulation failed.")
    else:
        st.info("No model available for prediction.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9AA0B4;font-size:.82rem;padding:12px 0;'>
  NVIDIA Price Predictor &nbsp;|&nbsp;
  Data: Yahoo Finance (OHLCV) · NewsAPI.org &nbsp;|&nbsp;
  Models: Ridge Regression · Monte Carlo GBM &nbsp;|&nbsp;
  NLP: VADER · FinBERT (ProsusAI) &nbsp;|&nbsp;
  <b style='color:#76B900'>For educational purposes only. Not financial advice.</b>
</div>
""", unsafe_allow_html=True)

# Clear the run flag at end
if st.session_state.get("nvidia_run"):
    st.session_state["nvidia_run"] = False
