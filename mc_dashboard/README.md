# Monte Carlo Stock Price Simulator

Interactive Streamlit dashboard for probabilistic stock price forecasting using **Geometric Brownian Motion (GBM)** Monte Carlo simulation.

Supports all **333 NSE/BSE stocks** in the master universe (Nifty 50, Nifty Next 50, Midcap 50, sectoral indices, etc.).

---

## What it does

| Feature | Detail |
|---|---|
| **GBM Simulation** | 5 000 paths, vectorised NumPy — runs in ~15 ms |
| **Fan Chart** | Historical price + percentile bands (1%–99%) |
| **Return Distribution** | Histogram with VaR / CVaR markers |
| **Probability Table** | P(return > X%) for 12 thresholds |
| **Rolling Backtest** | 66+ windows, calibration curve, Interval Score |
| **Universe filter** | 26 index buckets including Nifty 50, Defence, Tourism, etc. |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/harshatangirala/nifty-beta-clustering.git
cd nifty-beta-clustering
```

### 2. Create virtual environment

```bash
# Windows
python -m venv mc_dashboard/.venv
mc_dashboard\.venv\Scripts\activate

# macOS / Linux
python3 -m venv mc_dashboard/.venv
source mc_dashboard/.venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r mc_dashboard/requirements.txt
```

### 4. Run

```bash
streamlit run mc_dashboard/app.py
```

Then open http://localhost:8501 in your browser.

---

## Deploying to Streamlit Cloud (shareable link)

1. Push to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Fill in:

| Field | Value |
|---|---|
| Repository | `harshatangirala/nifty-beta-clustering` |
| Branch | `main` |
| Main file path | `mc_dashboard/app.py` |

4. Click **Deploy** — Streamlit Cloud auto-detects `mc_dashboard/requirements.txt`.

---

## How the model works

### Geometric Brownian Motion

```
S(t + dt) = S(t) · exp( (μ - σ²/2)·dt + σ·√dt·Z )
```

- **μ** — mean daily log-return, estimated from your chosen calibration window
- **σ** — std dev of daily log-returns (daily volatility)
- **Z** — standard normal random shock
- The `σ²/2` term is the **Itô correction** (Jensen's inequality for exponentials)

All 5 000 paths are generated with a single `np.random.default_rng().standard_normal((n_sims, horizon))` call — no Python loops.

### What "accuracy" means for Monte Carlo

Monte Carlo is **not a point forecast** — it is a probability distribution.  
Accuracy = *calibration*: do stated confidence intervals actually contain the truth at the stated frequency?

| Metric | Interpretation |
|---|---|
| **Coverage @80% CI** | Should be ~80%. Too high → model is over-conservative. Too low → under-conservative. |
| **Coverage @95% CI** | Should be ~95%. |
| **Direction Accuracy** | % correct up/down calls. Baseline = 50% (coin flip). |
| **Median MAE** | Mean absolute difference between predicted median return and actual return. |
| **Interval Score** | Proper scoring rule. Rewards narrow intervals that still contain the outcome. Lower = better. |
| **Calibration Curve** | Plot nominal percentile vs actual fraction below. Perfect = 45° line. |

---

## Limitations

- GBM assumes **constant drift and volatility** — real markets have volatility clustering (GARCH effects) and fat tails.
- **Does not predict direction** — the model reflects historical drift. A stock with 15% p.a. drift will show P(profit) > 50% even if momentum has reversed.
- Use for **risk sizing and scenario analysis**, not entry/exit signals.
- Very short calibration windows (< 6 months) give noisy volatility estimates.
