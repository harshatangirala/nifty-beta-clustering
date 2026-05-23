import warnings
warnings.filterwarnings("ignore")

import os
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
START = "2023-01-01"
END   = "2026-04-01"
NIFTY = "^NSEI"

_CLUSTER_COLORS = {
    "Negative Beta": "#EF4444",
    "Neutral Beta":  "#F59E0B",
    "Positive Beta": "#10B981",
    "Low Beta":      "#60A5FA",
    "High Beta":     "#10B981",
    "Cluster 1":     "#EF4444",
    "Cluster 2":     "#F59E0B",
    "Cluster 3":     "#10B981",
    "Cluster 4":     "#A78BFA",
    "Cluster 5":     "#F97316",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_DIR, "master_stock_universe.csv"))
    df["NSE_Symbol"] = df["Symbol"] + ".NS"
    return df


# ttl=86400 → re-download once per day on Streamlit Cloud; cache is shared
# across all concurrent users so only the first visitor pays the ~60 s cost.
@st.cache_data(show_spinner=False, ttl=86400)
def build_metrics(syms: tuple) -> pd.DataFrame:
    """
    Download adjusted close prices for all NSE symbols + Nifty 50,
    then compute Beta, Alpha, Volatility, Correlation and R² for each stock
    via OLS regression of daily returns against the Nifty 50.
    """
    tickers = list(syms) + [NIFTY]

    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex (field, ticker) for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw  # fallback – should not happen with 300+ tickers

    if NIFTY not in close.columns:
        # yfinance occasionally fails on cloud — caller shows a user-friendly error
        return pd.DataFrame(columns=["Symbol","Beta","Alpha","Volatility","Correlation","R2","P_Value"])

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


@st.cache_data(show_spinner=False)
def run_clustering(metrics: pd.DataFrame, n: int, feats: tuple) -> pd.DataFrame:
    """K-Means on scaled features; clusters ranked and named by mean Beta."""
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

    if n == 2:
        names = ["Low Beta", "High Beta"]
    elif n == 3:
        names = ["Negative Beta", "Neutral Beta", "Positive Beta"]
    else:
        names = [f"Cluster {i + 1}" for i in range(n)]

    df = df.copy()
    df["Cluster"] = [names[r_map[int(l)]] for l in labels]
    return df


# ── Layout helpers ────────────────────────────────────────────────────────────

def _vline(fig, x, dash, color):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1)


def render_scatter(df: pd.DataFrame) -> go.Figure:
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
        color_discrete_map=_CLUSTER_COLORS,
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


def render_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Beta", color="Cluster",
        barmode="overlay", nbins=40, opacity=0.72,
        color_discrete_map=_CLUSTER_COLORS,
        labels={"Beta": "β"},
        title="Beta distribution",
    )
    _vline(fig, 0, "dash", "rgba(200,200,200,0.3)")
    _vline(fig, 1, "dot",  "rgba(100,180,255,0.35)")
    fig.update_layout(height=230, showlegend=False, margin=dict(t=36, b=0))
    return fig


def render_pie(df: pd.DataFrame) -> go.Figure:
    pie = df.groupby("Cluster").size().reset_index(name="n")
    fig = px.pie(
        pie, values="n", names="Cluster",
        color="Cluster", color_discrete_map=_CLUSTER_COLORS,
        hole=0.45, title="Cluster share",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=230, margin=dict(t=36, b=0, l=0, r=0),
                      showlegend=False)
    return fig


def render_industry_bar(df: pd.DataFrame) -> go.Figure:
    d = df.groupby(["Industry", "Cluster"]).size().reset_index(name="n")
    fig = px.bar(
        d, x="Industry", y="n", color="Cluster",
        color_discrete_map=_CLUSTER_COLORS, barmode="stack",
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


def render_beta_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="Cluster", y="Beta", color="Cluster",
        color_discrete_map=_CLUSTER_COLORS,
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
    st.caption(
        "**333 NSE stocks** · "
        "**Period: 1 Jan 2023 – 1 Apr 2026** · "
        "**Benchmark: Nifty 50 (^NSEI)** · "
        "Beta computed via OLS regression of daily returns"
    )

    univ = load_universe()
    syms = tuple(univ["NSE_Symbol"].tolist())

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
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
            "Results are cached; subsequent runs are instant."
        )

    # ── Data pipeline ─────────────────────────────────────────────────────────
    with st.status("Loading data…", expanded=True) as sts:
        st.write("⏳ Downloading prices from Yahoo Finance…")
        metrics_raw = build_metrics(syms)

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
        clustered = run_clustering(metrics_raw, n_cl, feats)
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
        st.plotly_chart(render_scatter(view), use_container_width=True)

    with col_side:
        st.subheader("Cluster Share")
        st.plotly_chart(render_pie(view), use_container_width=True)
        st.subheader("Beta Distribution")
        st.plotly_chart(render_histogram(view), use_container_width=True)

    # ── Box plot ──────────────────────────────────────────────────────────────
    st.subheader("Beta Box Plot by Cluster")
    st.plotly_chart(render_beta_box(view), use_container_width=True)

    # ── Industry stacked bar ──────────────────────────────────────────────────
    st.subheader("Industry × Cluster Breakdown")
    st.plotly_chart(render_industry_bar(view), use_container_width=True)

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
