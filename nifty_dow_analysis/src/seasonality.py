"""Monthly and weekly seasonality analysis for the Nifty 50 index."""

import numpy as np
import pandas as pd

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEEK_OF_MONTH_ORDER = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]


def add_seasonality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds MonthName, WeekOfMonth (turn-of-month bucket), and ISOWeek columns."""
    df = df.copy()
    df["MonthName"] = df.index.month_name()
    df["WeekOfMonth"] = "Week " + (((df.index.day - 1) // 7) + 1).astype(str)
    df["ISOWeek"] = df.index.isocalendar().week.astype(int)
    return df


def _agg_returns(grouped) -> pd.DataFrame:
    summary = grouped.agg(
        trading_days="count",
        mean_return="mean",
        median_return="median",
        std_dev="std",
        best_day="max",
        worst_day="min",
    )
    summary["win_rate_pct"] = grouped.apply(lambda s: (s > 0).mean() * 100)
    summary["cumulative_return_pct"] = grouped.apply(lambda s: (np.prod(1 + s / 100) - 1) * 100)
    return summary.round(4)


def month_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month (Jan-Dec), aggregated across all years."""
    df = add_seasonality_columns(df)
    summary = _agg_returns(df.groupby("MonthName")["PctChange"])
    return summary.reindex(MONTH_ORDER)


def week_of_month_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per week-in-month bucket (turn-of-month effect)."""
    df = add_seasonality_columns(df)
    summary = _agg_returns(df.groupby("WeekOfMonth")["PctChange"])
    present = [w for w in WEEK_OF_MONTH_ORDER if w in summary.index]
    return summary.reindex(present)


def week_of_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ISO week number (1-53), aggregated across all years."""
    df = add_seasonality_columns(df)
    summary = _agg_returns(df.groupby("ISOWeek")["PctChange"])
    summary.index.name = "ISOWeek"
    return summary.sort_index()


def yearly_month_means(df: pd.DataFrame) -> pd.DataFrame:
    """Year x Month matrix of mean returns, for the heatmap."""
    df = add_seasonality_columns(df)
    pivot = df.pivot_table(index="Year", columns="MonthName", values="PctChange", aggfunc="mean")
    return pivot.reindex(columns=MONTH_ORDER).round(3)


def build_seasonality_report(month_sum: pd.DataFrame, wom_sum: pd.DataFrame, woy_sum: pd.DataFrame) -> str:
    best_month = month_sum["mean_return"].idxmax()
    worst_month = month_sum["mean_return"].idxmin()
    best_wom = wom_sum["mean_return"].idxmax()
    worst_wom = wom_sum["mean_return"].idxmin()
    best_woy = int(woy_sum["mean_return"].idxmax())
    worst_woy = int(woy_sum["mean_return"].idxmin())

    lines = [
        f"- **{best_month}** has historically been the strongest month "
        f"(avg {month_sum.loc[best_month, 'mean_return']:+.3f}%/day, win rate {month_sum.loc[best_month, 'win_rate_pct']:.1f}%, "
        f"compounded {month_sum.loc[best_month, 'cumulative_return_pct']:+.1f}% across every {best_month} in the sample).",
        f"- **{worst_month}** has historically been the weakest month "
        f"(avg {month_sum.loc[worst_month, 'mean_return']:+.3f}%/day, win rate {month_sum.loc[worst_month, 'win_rate_pct']:.1f}%, "
        f"compounded {month_sum.loc[worst_month, 'cumulative_return_pct']:+.1f}%).",
        f"- **Turn-of-month effect:** **{best_wom}** of the month has averaged the best daily return "
        f"({wom_sum.loc[best_wom, 'mean_return']:+.3f}%), while **{worst_wom}** has averaged the worst "
        f"({wom_sum.loc[worst_wom, 'mean_return']:+.3f}%).",
        f"- **Week-of-year:** ISO week **{best_woy}** has had the highest average daily return historically "
        f"({woy_sum.loc[best_woy, 'mean_return']:+.3f}%), while week **{worst_woy}** has had the lowest "
        f"({woy_sum.loc[worst_woy, 'mean_return']:+.3f}%). Each week-of-year bucket only has ~11-12 observations per "
        f"weekday-equivalent (about 11 years of data), so this is the noisiest cut in the whole study.",
        "",
        "Caveat: month-of-year and turn-of-month effects rest on more observations per bucket than the week-of-year "
        "cut and are the more statistically grounded of the three seasonality views here. As with the day-of-week "
        "result, none of these gaps are large enough to clear transaction costs as a standalone strategy — treat "
        "them as mild historical tilts, not signals.",
    ]
    return "\n".join(lines)
