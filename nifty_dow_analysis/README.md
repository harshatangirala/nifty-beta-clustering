# Nifty 50 Seasonality Study

Two-layer seasonality dashboard for the Nifty 50 index (`^NSEI`), built to answer:
"which day of the week, which month, and which week is historically better to invest in the Indian stock market?"

- **Layer 1 — Day of week:** average return, win rate, return distribution, and a year-by-year heatmap per weekday.
- **Layer 2 — Seasonality:** average return by calendar month, turn-of-month effect (week-in-month), and week-of-year (ISO week) effect.

Data: daily OHLC from Yahoo Finance via `yfinance`, 1 Jan 2015 – 16 Jun 2026. Returns are `% change` vs. the
previous trading day's close.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` (or pass `--server.port <port>`).

## Project structure

```
nifty_dow_analysis/
├── app.py              # Streamlit app — both layers, tabbed
├── src/
│   ├── data_fetcher.py # yfinance pull + local CSV cache
│   ├── analysis.py      # day-of-week return stats
│   └── seasonality.py   # month / week-of-month / week-of-year return stats
├── data/                # cached OHLC CSV (gitignored, regenerated on first run)
└── requirements.txt
```

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (already connected to `origin`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Pick this repository and branch (`main`).
4. Set **Main file path** to `nifty_dow_analysis/app.py`.
5. Streamlit Cloud auto-detects `nifty_dow_analysis/requirements.txt` since it lives next to the entry file.
6. Deploy — first load fetches and caches Nifty 50 data from Yahoo Finance automatically.

No secrets or API keys are required; the only external call is the public `yfinance` data pull.
