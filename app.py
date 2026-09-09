import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Resolve CSV path relative to this file so it works on any machine / cloud
_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Beta Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
NIFTY         = "^NSEI"
_DEFAULT_FROM = datetime.date(2023, 1, 1)
# NOTE: default "To" is computed dynamically at runtime (see main()), never a
# fixed calendar date — a hardcoded constant here would silently stop picking
# up new data the moment that date passed (this bit prior dashboards on this
# account, e.g. a fixed 2026-04-01 default is now ~5 months stale as of today).
_MIN_FROM     = datetime.date(2010, 1, 1)   # earliest allowed start

# Colors are assigned by cluster RANK (lowest mean beta -> red ... highest ->
# purple) at runtime in run_clustering(), not by a fixed name->color table.
# A fixed table keyed on label text broke whenever the label vocabulary
# changed or two clusters shared a label after dedup.
_RANK_PALETTE = ["#EF4444", "#F59E0B", "#10B981", "#60A5FA", "#A78BFA"]

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_DIR, "master_stock_universe.csv"))
    df["NSE_Symbol"] = df["Symbol"] + ".NS"
    return df


# ttl=86400 → re-download once per day on Streamlit Cloud; cache is keyed on
# (syms, start, end) so each unique date range gets its own cached result.
@st.cache_data(show_spinner=False, ttl=86400)
def build_metrics(syms: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all NSE symbols + Nifty 50,
    then compute Beta, Alpha, Volatility, Correlation and R² for each stock
    via OLS regression of daily returns against the Nifty 50.
    """
    tickers = list(syms) + [NIFTY]
    empty = pd.DataFrame(columns=["Symbol","Beta","Alpha","Volatility","Correlation","R2","P_Value"])

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        # No internet / Yahoo Finance outage / rate-limit — raised as a hard
        # exception by yfinance rather than an empty frame. Route it through
        # the same "empty" path so the caller's friendly error message (and
        # st.stop()) fires instead of an unhandled traceback crashing the app.
        return empty

    if raw is None or raw.empty:
        return empty

    # yfinance returns MultiIndex (field, ticker) columns for multi-ticker
    # downloads (this has changed across yfinance versions — handle both).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw  # fallback – should not happen with 300+ tickers

    if NIFTY not in close.columns:
        # yfinance occasionally fails on cloud — caller shows a user-friendly error
        return empty

    mkt = close[NIFTY].pct_change().dropna()
    rows = []

    for sym in syms:
        if sym not in close.columns:
            continue
        r   = close[sym].pct_change().dropna()
        idx = mkt.index.intersection(r.index)
        if len(idx) < 120:
            continue

        x = mkt.loc[idx].values.astype(float)
        y = r.loc[idx].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 120:
            continue
        x, y = x[ok], y[ok]

        b, a, rv, pv, _ = stats.linregress(x, y)
        rows.append({
            "Symbol":      sym.replace(".NS", ""),
            "Beta":        round(b, 4),
            "Alpha":       round(a * 252 * 100, 2),          # annualised %
            "Volatility":  round(np.std(y) * np.sqrt(252) * 100, 2),  # ann %
            "Correlation": round(rv, 4),
            "R2":          round(rv * rv, 4),
            "P_Value":     round(pv, 4),
        })

    return pd.DataFrame(rows)


# Beta buckets used to label a cluster from its ACTUAL mean beta, rather than
# from its rank alone. A pure rank->name table (e.g. "lowest-beta cluster is
# always called Negative Beta") mislabels data whenever the real distribution
# doesn't span both signs — e.g. on the Nifty 500 universe, where almost every
# stock has a positive beta, the lowest-beta cluster still had a positive mean
# but was shown as "Negative Beta" (verified: a 15-stock test run produced a
# "Negative Beta" cluster containing only stocks with beta 0.46-0.72).
_BETA_BUCKETS = [
    (-float("inf"), -0.05, "Negative Beta"),
    (-0.05,          0.85, "Low Beta"),
    (0.85,           1.15, "Market Beta"),
    (1.15,   float("inf"), "High Beta"),
]


def _bucket_label(mean_beta: float) -> str:
    for lo, hi, label in _BETA_BUCKETS:
        if lo <= mean_beta < hi:
            return label
    return "High Beta"


@st.cache_data(show_spinner=False)
def run_clustering(metrics: pd.DataFrame, n: int, feats: tuple):
    """K-Means on scaled features; clusters ranked and labeled by mean Beta.

    Returns (dataframe_with_cluster_column, color_map) where color_map maps
    each cluster label actually produced this run to a color, assigned by
    rank (lowest mean beta -> red ... highest -> purple).
    """
    feats = list(feats)
    X  = metrics[feats].values.astype(float)
    ok = np.all(np.isfinite(X), axis=1)
    df = metrics[ok].copy()
    X  = X[ok]

    Xs     = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n, random_state=42, n_init=20, max_iter=500).fit_predict(Xs)

    # Rank clusters by mean Beta (ascending → index 0 = lowest beta)
    b_idx = feats.index("Beta")
    means = [X[labels == i, b_idx].mean() for i in range(n)]
    order = np.argsort(means)
    r_map = {int(c): rank for rank, c in enumerate(order)}

    # Label each rank from its actual mean beta, then disambiguate duplicates
    # (e.g. two different clusters both landing in the "High Beta" bucket) so
    # distinct KMeans clusters never silently collapse into one legend entry.
    raw_names = [_bucket_label(means[c]) for c in order]
    seen: dict = {}
    names = []
    for nm in raw_names:
        seen[nm] = seen.get(nm, 0) + 1
        names.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")

    positions = np.linspace(0, len(_RANK_PALETTE) - 1, n).round().astype(int)
    color_map = {name: _RANK_PALETTE[pos] for name, pos in zip(names, positions)}

    df = df.copy()
    df["Cluster"] = [names[r_map[int(l)]] for l in labels]
    return df, color_map


# ── Layout helpers ────────────────────────────────────────────────────────────

def _vline(fig, x, dash, color):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1)


def render_scatter(df: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = px.scatter(
        df,
        x="Beta",
        y="Volatility",
        color="Cluster",
        hover_data={
            "Company Name": True,
            "Symbol":       True,
            "Industry":     True,
            "Beta":         ":.4f",
            "Volatility":   ":.2f",
            "Correlation":  ":.3f",
            "R2":           ":.3f",
            "Cluster":      False,
        },
        color_discrete_map=color_map,
        labels={"Beta": "β  (vs Nifty 50)", "Volatility": "Ann. Volatility (%)"},
        title=f"Beta vs Volatility — {len(df)} stocks",
    )
    _vline(fig, 0, "dash", "rgba(200,200,200,0.3)")
    _vline(fig, 1, "dot",  "rgba(100,180,255,0.35)")
    fig.add_annotation(x=0.01, y=df["Volatility"].quantile(0.97),
                       text="β=0", showarrow=False,
                       font=dict(color="gray", size=11))
    fig.add_annotation(x=1.01, y=df["Volatility"].quantile(0.97),
                       text="β=1", showarrow=False,
                       font=dict(color="cornflowerblue", size=11))
    fig.update_layout(height=480, legend=dict(orientation="h", y=1.06, x=0))
    return fig


def render_histogram(df: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = px.histogram(
        df, x="Beta", color="Cluster",
        barmode="overlay", nbins=40, opacity=0.72,
        color_discrete_map=color_map,
        labels={"Beta": "β"},
        title="Beta distribution",
    )
    _vline(fig, 0, "dash", "rgba(200,200,200,0.3)")
    _vline(fig, 1, "dot",  "rgba(100,180,255,0.35)")
    fig.update_layout(height=230, showlegend=False, margin=dict(t=36, b=0))
    return fig


def render_pie(df: pd.DataFrame, color_map: dict) -> go.Figure:
    pie = df.groupby("Cluster").size().reset_index(name="n")
    fig = px.pie(
        pie, values="n", names="Cluster",
        color="Cluster", color_discrete_map=color_map,
        hole=0.45, title="Cluster share",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=230, margin=dict(t=36, b=0, l=0, r=0),
                      showlegend=False)
    return fig


def render_industry_bar(df: pd.DataFrame, color_map: dict) -> go.Figure:
    d = df.groupby(["Industry", "Cluster"]).size().reset_index(name="n")
    fig = px.bar(
        d, x="Industry", y="n", color="Cluster",
        color_discrete_map=color_map, barmode="stack",
        labels={"n": "Stocks"},
        title="Stock count by Industry and Beta Cluster",
    )
    fig.update_xaxes(tickangle=45, tickfont_size=10)
    fig.update_layout(height=420, legend=dict(orientation="h", y=1.05))
    return fig


def render_index_bar(df: pd.DataFrame) -> go.Figure:
    d = (df.groupby("Index Name")
           .agg(Avg_Beta=("Beta", "mean"), Count=("Beta", "count"))
           .reset_index()
           .sort_values("Avg_Beta"))
    fig = px.bar(
        d, x="Avg_Beta", y="Index Name", orientation="h",
        text=d["Count"].astype(str) + " stks",
        color="Avg_Beta",
        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
        labels={"Avg_Beta": "Avg β", "Index Name": ""},
        title="Average β by Index",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=480, coloraxis_showscale=False,
                      margin=dict(l=200))
    return fig


def render_beta_box(df: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = px.box(
        df, x="Cluster", y="Beta", color="Cluster",
        color_discrete_map=color_map,
        points="outliers",
        title="Beta distribution per cluster (box plot)",
        labels={"Beta": "β"},
    )
    _vline(fig, -0.5, "dash", "rgba(200,200,200,0.0)")  # dummy to keep style
    fig.update_layout(height=380, showlegend=False)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Nifty Stock Beta Clustering Dashboard")

    univ = load_universe()
    syms = tuple(univ["NSE_Symbol"].tolist())
    today = datetime.date.today()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📅 Date Range")
        date_from = st.date_input(
            "From",
            value=_DEFAULT_FROM,
            min_value=_MIN_FROM,
            max_value=today - datetime.timedelta(days=180),
            format="DD/MM/YYYY",
        )
        date_to = st.date_input(
            "To",
            value=today,  # always defaults to the latest available date, not
                          # a fixed calendar date that would go stale over time
            min_value=date_from + datetime.timedelta(days=180),
            max_value=today,
            format="DD/MM/YYYY",
        )

        # Guard: ensure valid range before proceeding
        if date_to <= date_from:
            st.error("'To' date must be after 'From' date.")
            st.stop()

        start_str = date_from.strftime("%Y-%m-%d")
        end_str   = date_to.strftime("%Y-%m-%d")

        st.divider()
        st.header("⚙️ Clustering")
        n_cl = st.slider("Number of clusters", 2, 5, 3)

        extra = st.multiselect(
            "Extra features (Beta always included)",
            ["Volatility", "Alpha", "Correlation", "R2"],
            default=["Volatility"],
            help="Feature set used to compute K-Means distance.",
        )
        feats = tuple(["Beta"] + extra)

        st.divider()
        st.header("🔍 Filters")
        s_ind = st.selectbox("Industry", ["All"] + sorted(univ["Industry"].unique()))
        s_idx = st.selectbox("Index",    ["All"] + sorted(univ["Index Name"].unique()))

        st.divider()
        st.caption(
            "Data: Yahoo Finance via yfinance.  \n"
            "First load downloads ~334 tickers — allow ~60–90 s.  \n"
            "Results are cached per date range; subsequent runs are instant."
        )

    # Dynamic caption — placed here so date_from/date_to are already defined.
    # Stock count comes from len(syms) (the loaded universe), not a hardcoded
    # number, so it can't drift out of sync if master_stock_universe.csv changes.
    st.caption(
        f"**{len(syms)} NSE stocks** · "
        f"**Period: {date_from.strftime('%d %b %Y')} – {date_to.strftime('%d %b %Y')}** · "
        f"**Benchmark: Nifty 50 (^NSEI)** · "
        f"Beta computed via OLS regression of daily returns"
    )

    # ── Data pipeline ─────────────────────────────────────────────────────────
    with st.status("Loading data…", expanded=True) as sts:
        st.write("⏳ Downloading prices from Yahoo Finance…")
        metrics_raw = build_metrics(syms, start_str, end_str)

        if metrics_raw.empty:
            st.error(
                "❌ Could not download Nifty 50 (^NSEI) data from Yahoo Finance.  \n"
                "**Possible causes:** no internet access, Yahoo Finance rate-limit, or "
                "a temporary outage.  \n"
                "Wait a minute and hit **Rerun** (top-right ⋮ menu)."
            )
            st.stop()

        n_ok     = len(metrics_raw)
        n_failed = len(syms) - n_ok
        st.write(
            f"✅ Metrics computed for **{n_ok}** stocks"
            + (f" — {n_failed} skipped (< 120 trading days)" if n_failed else "")
        )

        st.write(f"🔄 Running K-Means (k={n_cl}, features: {', '.join(feats)})…")
        clustered, color_map = run_clustering(metrics_raw, n_cl, feats)
        clustered = clustered.merge(
            univ[["Symbol", "Company Name", "Industry", "Index Name"]],
            on="Symbol", how="left",
        )
        sts.update(label="✅ Ready", state="complete", expanded=False)

    # ── Filters ───────────────────────────────────────────────────────────────
    view = clustered.copy()
    if s_ind != "All":
        view = view[view["Industry"]   == s_ind]
    if s_idx != "All":
        view = view[view["Index Name"] == s_idx]

    # ── KPI row ───────────────────────────────────────────────────────────────
    k = st.columns(5)
    k[0].metric("Total Stocks Analyzed", len(clustered))
    k[1].metric("Stocks in Current View", len(view))
    k[2].metric("Avg β (all stocks)",  f"{clustered['Beta'].mean():.3f}")
    k[3].metric("Avg Volatility",      f"{clustered['Volatility'].mean():.1f}%")
    k[4].metric("Avg R²",              f"{clustered['R2'].mean():.3f}")

    st.divider()

    # ── Cluster summary table ─────────────────────────────────────────────────
    st.subheader("Cluster Summary")
    summ = (
        clustered.groupby("Cluster", sort=False)
        .agg(
            Count        = ("Beta", "count"),
            Beta_Avg     = ("Beta", "mean"),
            Beta_Median  = ("Beta", "median"),
            Beta_Std     = ("Beta", "std"),
            Vol_Avg      = ("Volatility", "mean"),
            Corr_Avg     = ("Correlation", "mean"),
            R2_Avg       = ("R2", "mean"),
            Alpha_Avg    = ("Alpha", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("Beta_Avg")
    )
    st.dataframe(summ, use_container_width=True)

    st.divider()

    # ── Scatter + Pie + Histogram ──────────────────────────────────────────────
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.subheader("Beta vs Volatility")
        st.plotly_chart(render_scatter(view, color_map), use_container_width=True)

    with col_side:
        st.subheader("Cluster Share")
        st.plotly_chart(render_pie(view, color_map), use_container_width=True)
        st.subheader("Beta Distribution")
        st.plotly_chart(render_histogram(view, color_map), use_container_width=True)

    # ── Box plot ──────────────────────────────────────────────────────────────
    st.subheader("Beta Box Plot by Cluster")
    st.plotly_chart(render_beta_box(view, color_map), use_container_width=True)

    # ── Industry stacked bar ──────────────────────────────────────────────────
    st.subheader("Industry × Cluster Breakdown")
    st.plotly_chart(render_industry_bar(view, color_map), use_container_width=True)

    # ── Index avg beta ────────────────────────────────────────────────────────
    st.subheader("Average β by Index")
    st.plotly_chart(render_index_bar(clustered), use_container_width=True)

    # ── Stock detail table ────────────────────────────────────────────────────
    st.subheader("All Stocks")

    t_col1, t_col2 = st.columns([3, 2])
    srch = t_col1.text_input("Search by name or symbol",
                              placeholder="e.g. Reliance, TCS, HDFC…")
    cl_f = t_col2.multiselect("Filter by cluster",
                               sorted(view["Cluster"].unique()), default=[])

    tbl = view[[
        "Company Name", "Symbol", "Industry", "Index Name",
        "Cluster", "Beta", "Alpha", "Volatility", "Correlation", "R2",
    ]].rename(columns={"Alpha": "Alpha (%/yr)", "Volatility": "Vol (%/yr)"})

    if srch:
        mask = (
            tbl["Company Name"].str.contains(srch, case=False, na=False) |
            tbl["Symbol"].str.contains(srch, case=False, na=False)
        )
        tbl = tbl[mask]
    if cl_f:
        tbl = tbl[tbl["Cluster"].isin(cl_f)]

    st.dataframe(
        tbl.sort_values("Beta", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=520,
        column_config={
            "Beta":           st.column_config.NumberColumn(format="%.4f"),
            "Alpha (%/yr)":   st.column_config.NumberColumn(format="%.2f"),
            "Vol (%/yr)":     st.column_config.NumberColumn(format="%.2f"),
            "Correlation":    st.column_config.NumberColumn(format="%.3f"),
            "R2":             st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.download_button(
        "⬇️ Download filtered data as CSV",
        tbl.to_csv(index=False),
        "nifty_beta_clusters.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
