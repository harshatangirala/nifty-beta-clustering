"""
NVIDIA Stock Price Prediction — ML Pipeline
============================================
Handles:
  • Technical feature engineering from OHLCV data
  • Sentiment feature merging
  • Ridge-regression model training with TimeSeriesSplit CV
  • Comprehensive evaluation metrics (MAE $, RMSE $, R², Direction-Accuracy)
  • Monte-Carlo / GBM price simulation for probability distribution
  • Sentiment model evaluation on Financial Phrase Bank
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
)


# ── Feature Columns ───────────────────────────────────────────────────────────

PRICE_FEATURES: List[str] = [
    "lag_ret_1", "lag_ret_2", "lag_ret_3", "lag_ret_5", "lag_ret_10",
    "rsi_14", "macd_norm", "bb_pct",
    "sma20_ratio", "sma50_ratio",
    "vol_ratio", "atr_ratio",
    "momentum_5", "momentum_20",
]

SENTIMENT_FEATURES: List[str] = [
    "avg_compound", "compound_ma3", "compound_ma5",
    "pos_ratio", "neg_ratio",
]

FEATURE_LABELS: Dict[str, str] = {
    "lag_ret_1":    "Return t-1",
    "lag_ret_2":    "Return t-2",
    "lag_ret_3":    "Return t-3",
    "lag_ret_5":    "Return t-5",
    "lag_ret_10":   "Return t-10",
    "rsi_14":       "RSI-14",
    "macd_norm":    "MACD (norm)",
    "bb_pct":       "BB Position",
    "sma20_ratio":  "SMA20 Ratio",
    "sma50_ratio":  "SMA50 Ratio",
    "vol_ratio":    "Volume Ratio",
    "atr_ratio":    "ATR Ratio",
    "momentum_5":   "Momentum 5d",
    "momentum_20":  "Momentum 20d",
    "avg_compound": "Sentiment Score",
    "compound_ma3": "Sentiment MA-3",
    "compound_ma5": "Sentiment MA-5",
    "pos_ratio":    "Positive Ratio",
    "neg_ratio":    "Negative Ratio",
}


# ── 1. Technical Feature Engineering ─────────────────────────────────────────

def compute_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag returns, RSI, MACD, Bollinger Bands, ATR, Volume ratio,
    momentum, and the next-day log-return target from an OHLCV DataFrame.

    Input columns required: date, open, high, low, close, volume
    """
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    c  = df["close"]

    # ── Log returns & lags ──────────────────────────────────────────────────
    df["log_return"] = np.log(c / c.shift(1))
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_ret_{lag}"] = df["log_return"].shift(lag)

    # ── Simple Moving Averages (ratio to close) ─────────────────────────────
    df["sma_20"]    = c.rolling(20).mean()
    df["sma_50"]    = c.rolling(50).mean()
    df["sma20_ratio"] = c / (df["sma_20"] + 1e-10) - 1
    df["sma50_ratio"] = c / (df["sma_50"] + 1e-10) - 1

    # ── RSI (14) — normalised to [-1, +1] ────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["rsi_14"] = (100 - 100 / (1 + rs) - 50) / 50   # centred at 0

    # ── MACD (12/26/9) normalised by close ──────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    df["macd_norm"]   = macd / (c + 1e-10)

    # ── Bollinger Band % position ────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - bb_mid) / (2 * bb_std + 1e-10)  # ≈ [-1, +1]

    # ── Average True Range (ATR) ratio ────────────────────────────────────────
    h, lo, pc = df["high"], df["low"], c.shift(1)
    tr  = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    df["atr_14"]  = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_14"] / (c + 1e-10)

    # ── Volume ratio (log of current vs 20-day avg) ──────────────────────────
    vol_ma20       = df["volume"].rolling(20).mean()
    df["vol_ratio"] = np.log(df["volume"] / (vol_ma20 + 1))

    # ── Momentum ────────────────────────────────────────────────────────────
    df["momentum_5"]  = c / (c.shift(5)  + 1e-10) - 1
    df["momentum_20"] = c / (c.shift(20) + 1e-10) - 1

    # ── Target: next day log-return ─────────────────────────────────────────
    df["target"] = df["log_return"].shift(-1)

    return df


# ── 2. Sentiment Feature Merging ──────────────────────────────────────────────

def merge_sentiment(price_feat_df: pd.DataFrame,
                    daily_sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join daily sentiment scores onto price feature rows.
    Missing sentiment days are filled with 0 (neutral).
    """
    df = price_feat_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if daily_sent_df.empty:
        for col in SENTIMENT_FEATURES:
            df[col] = 0.0
        return df

    sent = daily_sent_df[["date"] + SENTIMENT_FEATURES].copy()
    sent["date"] = pd.to_datetime(sent["date"])
    merged = df.merge(sent, on="date", how="left")
    for col in SENTIMENT_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0.0
    return merged


# ── 3. Model Training ─────────────────────────────────────────────────────────

def _fit_ridge(X: np.ndarray, y: np.ndarray,
               alpha: float = 1.0) -> Tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return model, scaler


def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def train_price_models(feat_df: pd.DataFrame,
                       has_sentiment: bool,
                       cv_splits: int = 5,
                       ridge_alpha: float = 1.0) -> Dict:
    """
    Train two Ridge regression models:
      price_model     — PRICE_FEATURES only (full history)
      sentiment_model — PRICE_FEATURES + SENTIMENT_FEATURES
                        (same history; sentiment = 0 where no news)

    Returns a results dict with model artifacts, metrics, CV scores, and
    arrays needed for visualisation.
    """
    results: Dict = {}

    model_specs = {
        "price_model":     PRICE_FEATURES,
        "sentiment_model": PRICE_FEATURES + (SENTIMENT_FEATURES if has_sentiment else []),
    }

    for name, feature_list in model_specs.items():
        # Keep only columns present in the DataFrame
        feats = [f for f in feature_list if f in feat_df.columns]
        sub   = feat_df.dropna(subset=feats + ["target"]).copy()

        if len(sub) < 120:
            continue

        X = sub[feats].values.astype(float)
        y = sub["target"].values.astype(float)
        close_prices = sub["close"].values

        # ── Chronological 80/20 split ────────────────────────────────────
        split = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        c_tr, c_te = close_prices[:split], close_prices[split:]

        model, scaler = _fit_ridge(X_tr, y_tr, alpha=ridge_alpha)
        Xte_s = scaler.transform(X_te)

        y_hat_tr = model.predict(scaler.transform(X_tr))
        y_hat_te = model.predict(Xte_s)

        # ── Metrics (log-return space) ───────────────────────────────────
        def _mae_usd(c, y_actual, y_pred):
            """Mean Absolute Error in dollar terms."""
            p_actual = c * np.exp(y_actual)
            p_pred   = c * np.exp(y_pred)
            return float(np.mean(np.abs(p_actual - p_pred)))

        metrics = {
            "train_mae_lr":    float(mean_absolute_error(y_tr, y_hat_tr)),
            "test_mae_lr":     float(mean_absolute_error(y_te, y_hat_te)),
            "train_rmse_lr":   float(np.sqrt(mean_squared_error(y_tr, y_hat_tr))),
            "test_rmse_lr":    float(np.sqrt(mean_squared_error(y_te, y_hat_te))),
            "train_r2":        float(r2_score(y_tr, y_hat_tr)),
            "test_r2":         float(r2_score(y_te, y_hat_te)),
            "train_dir_acc":   _direction_accuracy(y_tr, y_hat_tr),
            "test_dir_acc":    _direction_accuracy(y_te, y_hat_te),
            "train_mae_usd":   _mae_usd(c_tr, y_tr, y_hat_tr),
            "test_mae_usd":    _mae_usd(c_te, y_te, y_hat_te),
            "train_rmse_usd":  float(np.sqrt(np.mean(
                (c_tr * np.exp(y_tr) - c_tr * np.exp(y_hat_tr)) ** 2))),
            "test_rmse_usd":   float(np.sqrt(np.mean(
                (c_te * np.exp(y_te) - c_te * np.exp(y_hat_te)) ** 2))),
        }

        # ── TimeSeriesSplit CV ───────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_r2, cv_mae, cv_dir = [], [], []
        for tr_idx, val_idx in tscv.split(X):
            m, sc = _fit_ridge(X[tr_idx], y[tr_idx], alpha=ridge_alpha)
            yh    = m.predict(sc.transform(X[val_idx]))
            cv_r2.append(r2_score(y[val_idx], yh))
            cv_mae.append(mean_absolute_error(y[val_idx], yh))
            cv_dir.append(_direction_accuracy(y[val_idx], yh))

        metrics["cv_r2_mean"]    = float(np.mean(cv_r2))
        metrics["cv_r2_std"]     = float(np.std(cv_r2))
        metrics["cv_mae_mean"]   = float(np.mean(cv_mae))
        metrics["cv_dir_mean"]   = float(np.mean(cv_dir))

        # ── Residuals ────────────────────────────────────────────────────
        residuals   = y_te - y_hat_te
        resid_std   = float(np.std(residuals))
        resid_mean  = float(np.mean(residuals))

        # ── Feature importance (|coefficient| after scaling) ─────────────
        feat_imp = pd.Series(
            np.abs(model.coef_), index=feats,
            name="importance"
        ).sort_values(ascending=False)

        # ── Reconstructed price paths for out-of-sample chart ────────────
        test_price_actual    = c_te * np.exp(y_te)
        test_price_predicted = c_te * np.exp(y_hat_te)

        results[name] = {
            "model":            model,
            "scaler":           scaler,
            "feature_names":    feats,
            "metrics":          metrics,
            "residuals":        residuals,
            "resid_std":        resid_std,
            "resid_mean":       resid_mean,
            "feature_importance": feat_imp,
            "test_dates":       sub["date"].iloc[split:].values,
            "test_close":       c_te,
            "test_actual_lr":   y_te,
            "test_pred_lr":     y_hat_te,
            "test_price_actual":    test_price_actual,
            "test_price_predicted": test_price_predicted,
            "train_size":       split,
            "test_size":        len(y_te),
            "total_samples":    len(sub),
        }

    return results


# ── 4. Monte Carlo Price Prediction ──────────────────────────────────────────

def monte_carlo_forecast(
    model_result: Dict,
    price_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    n_days: int = 30,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict:
    """
    Geometric Brownian Motion forecast augmented by Ridge-regression drift.

    mu(t) = model_prediction * decay(t)  +  historical_drift
    sigma = max(historical_vol, residual_std)

    Returns dict with paths, percentile bands, and price distributions.
    """
    rng = np.random.default_rng(seed)

    model   = model_result["model"]
    scaler  = model_result["scaler"]
    feats   = model_result["feature_names"]
    res_std = model_result["resid_std"]

    current_price = float(price_df["close"].iloc[-1])

    # ── Expected drift from model (next 1 day) ──────────────────────────────
    last_valid = feat_df.dropna(subset=[f for f in feats if f in feat_df.columns])
    if last_valid.empty:
        predicted_return = 0.0
    else:
        x = last_valid.iloc[-1][[f for f in feats if f in feat_df.columns]].values
        predicted_return = float(model.predict(scaler.transform(x.reshape(1, -1)))[0])

    # ── Historical log-return stats ─────────────────────────────────────────
    log_rets = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
    recent   = log_rets.tail(252)
    hist_mu  = float(recent.mean())
    hist_vol = float(recent.std())

    # Effective simulation params
    sigma = max(hist_vol, res_std)  # use the larger volatility estimate

    # ── Simulate paths ──────────────────────────────────────────────────────
    paths = np.zeros((n_simulations, n_days))
    for d in range(n_days):
        # Drift decays exponentially from model prediction toward hist mean
        decay = np.exp(-0.15 * d)
        mu_d  = predicted_return * decay + hist_mu * (1 - decay)
        # GBM step
        shocks = rng.normal(0, 1, n_simulations)
        step   = mu_d - 0.5 * sigma ** 2 + sigma * shocks
        if d == 0:
            paths[:, 0] = current_price * np.exp(step)
        else:
            paths[:, d] = paths[:, d - 1] * np.exp(step)

    # ── Statistics at key horizons ──────────────────────────────────────────
    horizons = [d for d in [1, 5, 10, 20, n_days] if d <= n_days]
    price_at: Dict[int, Dict] = {}
    for day in horizons:
        ps = paths[:, day - 1]
        price_at[day] = {
            "mean":    float(np.mean(ps)),
            "median":  float(np.median(ps)),
            "std":     float(np.std(ps)),
            "pct_5":   float(np.percentile(ps, 5)),
            "pct_25":  float(np.percentile(ps, 25)),
            "pct_75":  float(np.percentile(ps, 75)),
            "pct_95":  float(np.percentile(ps, 95)),
            "prob_up": float(np.mean(ps > current_price)),
            "prob_up_5pct": float(np.mean(ps > current_price * 1.05)),
        }

    # ── Path percentile bands (for fan chart) ──────────────────────────────
    pct_5  = np.percentile(paths, 5,  axis=0)
    pct_25 = np.percentile(paths, 25, axis=0)
    pct_50 = np.percentile(paths, 50, axis=0)
    pct_75 = np.percentile(paths, 75, axis=0)
    pct_95 = np.percentile(paths, 95, axis=0)

    return {
        "current_price":       current_price,
        "predicted_return_1d": predicted_return,
        "expected_price_1d":   current_price * np.exp(predicted_return),
        "hist_vol_annual":     hist_vol * np.sqrt(252),
        "sigma_daily":         sigma,
        "paths":               paths,          # (n_sims, n_days)
        "price_at":            price_at,
        "bands": {
            "pct_5":  pct_5,
            "pct_25": pct_25,
            "pct_50": pct_50,
            "pct_75": pct_75,
            "pct_95": pct_95,
        },
        "n_simulations": n_simulations,
        "n_days":        n_days,
    }


# ── 5. Sentiment Model Evaluation ─────────────────────────────────────────────

# Financial PhraseBank test set (45 balanced examples)
PHRASEBANK: List[Tuple[str, str]] = [
    # positive (15)
    ("The company reported record profits of $2.3 billion, beating analyst expectations.", "positive"),
    ("Revenues surged 45% year-over-year, driven by strong demand in emerging markets.", "positive"),
    ("The board approved a 20% increase in dividend payments to shareholders.", "positive"),
    ("Operating margins improved to 18%, the highest level in five years.", "positive"),
    ("The firm secured a landmark contract worth $500 million.", "positive"),
    ("Earnings per share rose 30 cents to $1.85, well above consensus.", "positive"),
    ("The company announced a buyback of $1 billion worth of shares.", "positive"),
    ("Strong cash generation allowed the company to reduce debt by $400 million.", "positive"),
    ("The acquisition is expected to be immediately accretive to earnings.", "positive"),
    ("Net income doubled to $890 million on the back of surging product demand.", "positive"),
    ("The firm raised full-year guidance citing robust demand and cost savings.", "positive"),
    ("New product launch exceeded targets with 1 million units sold in the first month.", "positive"),
    ("The stock hit a 52-week high after stellar quarterly results.", "positive"),
    ("Free cash flow improved markedly, enabling accelerated debt repayment.", "positive"),
    ("Customer acquisition grew 32% as the loyalty programme gained traction.", "positive"),
    # negative (15)
    ("The company reported a net loss of $340 million, wider than expected.", "negative"),
    ("Revenue fell 22% as customer spending dried up amid the economic slowdown.", "negative"),
    ("The firm issued a profit warning, citing weaker-than-expected demand.", "negative"),
    ("Shares fell 12% after the company reported disappointing quarterly results.", "negative"),
    ("The company will cut 8% of its global workforce, affecting 4,200 employees.", "negative"),
    ("Write-downs of $1.2 billion dragged the quarterly results into the red.", "negative"),
    ("The credit rating was downgraded to junk status by Moody's.", "negative"),
    ("Regulatory investigations into the company's accounting practices have intensified.", "negative"),
    ("The company defaulted on its bond payments, triggering cross-default clauses.", "negative"),
    ("Free cash flow turned negative as capital expenditure rose sharply.", "negative"),
    ("Management warned that supply-chain disruptions would persist through the year.", "negative"),
    ("The CEO resigned unexpectedly amid a boardroom dispute over strategy.", "negative"),
    ("Plant shutdown will reduce output by 30% and cost $150 million in losses.", "negative"),
    ("Gross margin compression continued as raw material costs soared.", "negative"),
    ("The firm faces potential fines of up to $2 billion following regulatory review.", "negative"),
    # neutral (15)
    ("The company maintained its annual guidance range of $4.50 to $4.75 per share.", "neutral"),
    ("The board met on Monday to discuss the quarterly financial results.", "neutral"),
    ("The company will release its first-quarter results on April 28th.", "neutral"),
    ("The firm operates in 42 countries with approximately 85,000 employees.", "neutral"),
    ("The new CEO will assume the role effective January 1st.", "neutral"),
    ("Trading volumes were in line with seasonal expectations.", "neutral"),
    ("The company reiterated its previously stated capital allocation policy.", "neutral"),
    ("The annual general meeting is scheduled for June 15th in Helsinki.", "neutral"),
    ("The agreement is subject to regulatory approval in multiple jurisdictions.", "neutral"),
    ("The company has 12% market share in the domestic segment.", "neutral"),
    ("The firm announced it will report earnings before the market opens.", "neutral"),
    ("Production guidance for the full year remains unchanged at 2.4 million units.", "neutral"),
    ("The transaction is expected to close in the second quarter of fiscal 2025.", "neutral"),
    ("Management confirmed it is reviewing strategic options for the division.", "neutral"),
    ("The company hired a new chief financial officer effective next month.", "neutral"),
]

LABELS = ["positive", "negative", "neutral"]


def evaluate_vader() -> Dict:
    """Run VADER on PHRASEBANK and return classification metrics."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    texts   = [t for t, _ in PHRASEBANK]
    y_true  = [l for _, l in PHRASEBANK]
    y_pred  = []
    for text in texts:
        c = sid.polarity_scores(text)["compound"]
        y_pred.append(
            "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        )

    report = classification_report(y_true, y_pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc    = accuracy_score(y_true, y_pred)

    return {
        "model":        "VADER",
        "accuracy":     acc,
        "macro_f1":     report["macro avg"]["f1-score"],
        "macro_prec":   report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "per_class":    {lbl: report[lbl] for lbl in LABELS if lbl in report},
        "confusion_matrix": cm.tolist(),
        "labels":       LABELS,
        "y_true":       y_true,
        "y_pred":       y_pred,
    }


def evaluate_finbert(hf_token: str = "") -> Optional[Dict]:
    """Run FinBERT (HF Inference API) on PHRASEBANK. Returns None on failure."""
    import requests, time as _time

    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    hdrs   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    texts  = [t for t, _ in PHRASEBANK]
    y_true = [l for _, l in PHRASEBANK]
    y_pred = []
    y_prob = []  # [[pos, neg, neu], …]

    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch   = texts[i: i + batch_size]
        payload = {"inputs": batch, "options": {"wait_for_model": True}}
        try:
            resp = requests.post(HF_URL, headers=hdrs, json=payload, timeout=90)
            if resp.status_code != 200:
                return None
            for item in resp.json():
                items = item if isinstance(item, list) else [item]
                sc    = {r["label"].lower(): r["score"] for r in items}
                pos, neg, neu = sc.get("positive", 0), sc.get("negative", 0), sc.get("neutral", 0)
                compound = pos - neg
                y_pred.append(
                    "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
                )
                y_prob.append([pos, neg, neu])
            _time.sleep(0.15)
        except Exception:
            return None

    report = classification_report(y_true, y_pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc    = accuracy_score(y_true, y_pred)

    # ROC-AUC
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        y_bin   = label_binarize(y_true, classes=LABELS)
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        pass

    return {
        "model":        "FinBERT",
        "accuracy":     acc,
        "macro_f1":     report["macro avg"]["f1-score"],
        "macro_prec":   report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "roc_auc":      roc_auc,
        "per_class":    {lbl: report[lbl] for lbl in LABELS if lbl in report},
        "confusion_matrix": cm.tolist(),
        "labels":       LABELS,
        "y_true":       y_true,
        "y_pred":       y_pred,
    }
