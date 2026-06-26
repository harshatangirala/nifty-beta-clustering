"""Day-of-week return analysis for the Nifty 50 index."""

import numpy as np
import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def compute_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Adds pct_change (vs previous trading day's close), Weekday, Year, Month columns."""
    df = ohlc.sort_index().copy()
    df["PrevClose"] = df["Close"].shift(1)
    df["PctChange"] = (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100
    df = df.dropna(subset=["PctChange"])
    df["Weekday"] = df.index.day_name()
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df = df[df["Weekday"].isin(WEEKDAY_ORDER)]  # NSE has no weekend sessions, but guard anyway
    return df


def weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per weekday: count, mean/median/std return, win rate, best/worst day."""
    grouped = df.groupby("Weekday")["PctChange"]
    summary = grouped.agg(
        trading_days="count",
        mean_return="mean",
        median_return="median",
        std_dev="std",
        best_day="max",
        worst_day="min",
    )
    summary["win_rate_pct"] = grouped.apply(lambda s: (s > 0).mean() * 100)
    summary["cumulative_return_pct"] = grouped.apply(
        lambda s: (np.prod(1 + s / 100) - 1) * 100
    )
    summary = summary.reindex(WEEKDAY_ORDER)
    return summary.round(4)


def yearly_weekday_means(df: pd.DataFrame) -> pd.DataFrame:
    """Year x Weekday matrix of mean returns, for the heatmap."""
    pivot = df.pivot_table(index="Year", columns="Weekday", values="PctChange", aggfunc="mean")
    return pivot.reindex(columns=WEEKDAY_ORDER).round(3)


def overall_stats(df: pd.DataFrame) -> dict:
    return {
        "start_date": df.index.min().date(),
        "end_date": df.index.max().date(),
        "total_trading_days": len(df),
        "overall_mean_return": round(df["PctChange"].mean(), 4),
        "overall_win_rate": round((df["PctChange"] > 0).mean() * 100, 2),
    }


def build_report(summary: pd.DataFrame) -> str:
    """Plain-language summary of the day-of-week effect."""
    best_mean = summary["mean_return"].idxmax()
    worst_mean = summary["mean_return"].idxmin()
    highest_win = summary["win_rate_pct"].idxmax()
    lowest_win = summary["win_rate_pct"].idxmin()
    most_volatile = summary["std_dev"].idxmax()
    least_volatile = summary["std_dev"].idxmin()
    cheapest_entry = summary["mean_return"].idxmin()

    lines = [
        f"- **{best_mean}** has the highest average daily return ({summary.loc[best_mean, 'mean_return']:+.3f}%) "
        f"and a win rate of {summary.loc[best_mean, 'win_rate_pct']:.1f}%.",
        f"- **{worst_mean}** has the lowest average daily return ({summary.loc[worst_mean, 'mean_return']:+.3f}%) "
        f"and a win rate of {summary.loc[worst_mean, 'win_rate_pct']:.1f}%.",
        f"- **{highest_win}** has the highest win rate ({summary.loc[highest_win, 'win_rate_pct']:.1f}% of sessions closed up).",
        f"- **{lowest_win}** has the lowest win rate ({summary.loc[lowest_win, 'win_rate_pct']:.1f}% of sessions closed up).",
        f"- **{most_volatile}** is the most volatile day (std dev {summary.loc[most_volatile, 'std_dev']:.3f}%); "
        f"**{least_volatile}** is the calmest (std dev {summary.loc[least_volatile, 'std_dev']:.3f}%).",
        "",
        f"**Buy-the-dip framing:** if you read 'better to invest' as 'cheapest average entry point relative to the "
        f"prior close', **{cheapest_entry}** has historically had the weakest average session "
        f"({summary.loc[cheapest_entry, 'mean_return']:+.3f}%), which is the closest thing to a recurring dip.",
        f"**Momentum framing:** if you read 'better to invest' as 'which day has historically rewarded holders the most', "
        f"**{best_mean}** wins, both on average return and cumulative compounded return "
        f"({summary.loc[best_mean, 'cumulative_return_pct']:+.1f}% compounded over the period from {best_mean} sessions alone).",
        "",
        "Caveat: differences between weekdays are small relative to day-to-day volatility (std dev columns are far "
        "larger than the mean differences), so this is a mild statistical tilt, not a reliable trading signal. "
        "Transaction costs, taxes, and the fact that you can only act on the open/close you actually observe would "
        "likely erode any edge here.",
    ]
    return "\n".join(lines)
