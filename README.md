# Nifty Beta Clustering

A Streamlit dashboard that computes each NSE-listed stock's beta against the Nifty 50 and groups stocks into clusters (Low/High Beta, or Negative/Low/Market/High Beta) using K-Means, so you can see at a glance which stocks move with the market, against it, or more/less violently than it.

## What this app does

1. **Loads the stock universe** — `master_stock_universe.csv` lists 333 NSE-listed stocks with Company Name, Industry, Index Name and Symbol. The app appends `.NS` to each symbol for Yahoo Finance.
2. **Downloads prices** — `yf.download()` pulls adjusted daily closes for all 333 stocks plus the Nifty 50 index (`^NSEI`) over a user-selected date range (default: 2023-01-01 to today). Multi-ticker downloads come back with `MultiIndex` columns (`(field, ticker)`); the app selects the `"Close"` level explicitly rather than assuming a flat column layout.
3. **Computes daily returns and beta** — for each stock, daily percentage returns are computed and date-aligned against the Nifty's daily returns via `Index.intersection()` (not positional slicing, so gaps from new listings, trading halts, or delistings on one side don't silently shift the other series). Non-finite values are masked out before fitting. Stocks with fewer than 120 aligned trading days are dropped. Beta, alpha, R², correlation and p-value are then computed via `scipy.stats.linregress(market_returns, stock_returns)` — i.e. proper OLS regression, not a plug-in `cov/var` formula — with the **slope** as beta and the **intercept** annualized as alpha.
4. **Clusters stocks with K-Means** — features (Beta plus any of Volatility/Alpha/Correlation/R² the user selects) are passed through `StandardScaler` before `KMeans`, so clustering isn't dominated by whichever feature happens to have the largest raw numeric range. The resulting clusters are ranked by mean beta and labeled from **actual** centroid values (Negative / Low / Market / High Beta), not from a fixed position-based name table.
5. **Renders results with Plotly** — a beta-vs-volatility scatter, a beta histogram, a cluster-share pie, a beta box plot per cluster, an industry × cluster stacked bar, an average-beta-by-index bar, and a searchable/filterable/downloadable stock table.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First load downloads price history for ~333 tickers, which takes roughly 60–90 seconds. Subsequent runs are served from `st.cache_data` (keyed on the selected symbols + date range, `ttl=86400`) and are near-instant until the cache expires or the date range changes.

## What was fixed in this pass

This repo used to share space with several unrelated bolted-on Streamlit apps (a DCF model, a Monte Carlo simulator, a news-sentiment dashboard, a seasonality study, a personal study tracker). Those have been split into their own repositories, leaving this repo with only the original beta-clustering dashboard. During that split, a full correctness review of `app.py` turned up two real bugs, both now fixed:

- **Cluster labels didn't reflect the actual data, only rank order.** `run_clustering()` used to assign labels from a fixed table by rank alone — e.g. for `n=3`, the lowest-mean-beta cluster was *always* called "Negative Beta", even if every stock in it had a positive beta. This is exactly what happens on this universe in practice: most Nifty 500 constituents have positive beta, so the "low" cluster is rarely actually negative. Verified with a live run on a 15-stock sample: the lowest cluster (mean beta ≈ 0.6, range 0.46–0.72, all positive) was being labeled **"Negative Beta"** — a materially misleading claim about the data. Fixed by labeling each cluster from its own actual mean beta against fixed thresholds (`< -0.05` → Negative, `< 0.85` → Low, `0.85–1.15` → Market, `> 1.15` → High), with a numeric suffix appended only when two different K-Means clusters land in the same bucket (so distinct clusters never silently collapse into one legend entry). Colors are now assigned by rank position rather than by label text, since the label vocabulary is no longer fixed-size.
- **Hardcoded, silently-expiring default end date.** `_DEFAULT_TO = datetime.date(2026, 4, 1)` was a fixed constant. As of this review (September 2026) that date is already ~5 months in the past, so the "To" date picker defaulted to a stale cutoff and the dashboard would never show newer data unless a user manually moved the date — a bug pattern that has previously bitten other dashboards on this account. Fixed by defaulting the "To" date to `datetime.date.today()`, computed at runtime on every run.

Also hardened while in the code (not correctness bugs in the sense of wrong numbers, but real defects):
- `build_metrics()` had no `try/except` around `yf.download()` — a network failure or Yahoo Finance outage would raise an uncaught exception and crash the whole app with a traceback, even though the surrounding code already had a friendly "could not download data" message ready for the *empty-result* case. A connectivity failure now routes through that same friendly-error path instead of crashing.
- The "**333 NSE stocks**" caption was a hardcoded literal, not derived from the loaded universe, so it would silently go stale if `master_stock_universe.csv` were ever edited. It now reads `len(syms)` from the actually-loaded data.

**Verified correct (no fix needed), specifically checked because these are common failure modes in this class of app:**
- **Beta formula**: confirmed via a standalone synthetic test (a stock series built as an exact, zero-noise 1.5×/−0.8× copy of a synthetic "market" series correctly recovers beta = 1.5 / −0.8 via `linregress`; a noisy 1.2×-beta series recovers ≈1.19 with R² ≈ 0.97).
- **Return date-alignment**: confirmed via the same synthetic test with an artificial 20-trading-day gap injected into the stock series (simulating a listing gap/halt) that the market series doesn't have — beta still comes out within ~1% of the true value, because alignment is done by `Index.intersection()` (date-based), not by row position.
- **StandardScaler before KMeans**: confirmed present in `run_clustering()` — features are scaled before clustering, so a raw-magnitude feature like Volatility (tens) doesn't dominate distance over Beta (single digits).
- **MultiIndex yfinance columns**: confirmed handled — `raw["Close"]` is selected explicitly when `yf.download()` returns `MultiIndex` columns for a multi-ticker call.
- **Cache correctness**: confirmed `build_metrics()`'s cache key is `(syms, start, end)` — a tuple of symbols plus both date strings — so different date ranges or universes correctly produce separate cache entries rather than serving stale data.

## Caveats

- **Beta over a fixed lookback window is not stable or predictive.** Betas here are estimated from whatever date range is selected; they will shift (sometimes substantially) if you change the window, and a historical beta is not a forecast of future co-movement with the market.
- **Cluster count and thresholds are heuristic, not statistically validated.** The number of K-Means clusters is a user-adjustable slider (2–5), and the beta bucket thresholds used for labeling (−0.05, 0.85, 1.15) are reasonable round-number cutoffs, not derived from any formal test — treat cluster boundaries as a visualization aid, not a rigorous classification.
- **K-Means with `k` fixed by the user, not chosen by the data.** No elbow-method / silhouette-score check is performed; a different `k` can produce a materially different grouping of the same stocks.
- **Small-sample stocks are silently dropped**, not flagged individually — any stock with fewer than 120 aligned trading days in the selected window (very recent listings, long trading halts) is excluded from the whole dashboard for that window.
- **Data quality is whatever Yahoo Finance returns** via `yfinance` — dividend/split adjustments, occasional gaps, and delisted-symbol errors are Yahoo's, not corrected here beyond the NaN/short-history filtering already described.
- This is not investment advice. It is a descriptive/exploratory tool for looking at historical co-movement with the Nifty 50, nothing more.
