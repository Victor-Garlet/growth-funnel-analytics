from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#475569"
GRID = "#E2E8F0"
BLUE = "#1D4ED8"
SLATE = "#64748B"
GREEN = "#059669"
ORANGE = "#EA580C"
TEAL = "#0F766E"


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root (README.md not found).")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
        }
    )
    sns.set_theme(style="whitegrid")


def build_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET temp_directory='data/processed/tmp';")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='2GB';")
    con.execute(
        """
        CREATE OR REPLACE VIEW events AS
        SELECT * FROM read_parquet([
          'data/processed/events_2019_oct.parquet',
          'data/processed/events_2019_nov.parquet'
        ]);
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW events_enriched AS
        SELECT
          *,
          DATE(event_time) AS event_date,
          STRFTIME(event_time, '%Y-%m') AS event_month,
          DATE_TRUNC('week', event_time) AS event_week_start
        FROM events;
        """
    )
    return con


def query_funnel(con: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    funnel = con.execute(
        """
        WITH user_steps AS (
          SELECT
            event_month,
            user_id,
            MAX(CASE WHEN event_type='view' THEN 1 ELSE 0 END) AS did_view,
            MAX(CASE WHEN event_type='cart' THEN 1 ELSE 0 END) AS did_cart,
            MAX(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS did_purchase
          FROM events_enriched
          GROUP BY event_month, user_id
        )
        SELECT
          event_month,
          SUM(did_view) AS view_users,
          SUM(CASE WHEN did_view=1 AND did_cart=1 THEN 1 ELSE 0 END) AS cart_users,
          SUM(CASE WHEN did_cart=1 AND did_purchase=1 THEN 1 ELSE 0 END) AS purchase_users,
          SUM(CASE WHEN did_view=1 AND did_cart=1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(did_view), 0) AS view_to_cart_rate,
          SUM(CASE WHEN did_cart=1 AND did_purchase=1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN did_view=1 AND did_cart=1 THEN 1 ELSE 0 END), 0) AS cart_to_purchase_rate,
          SUM(CASE WHEN did_view=1 AND did_purchase=1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(did_view), 0) AS view_to_purchase_rate
        FROM user_steps
        GROUP BY event_month
        ORDER BY event_month;
        """
    ).fetchdf()

    tracking_gap = con.execute(
        """
        WITH user_steps AS (
          SELECT
            event_month,
            user_id,
            MAX(CASE WHEN event_type='cart' THEN 1 ELSE 0 END) AS did_cart,
            MAX(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS did_purchase
          FROM events_enriched
          GROUP BY event_month, user_id
        )
        SELECT
          event_month,
          SUM(CASE WHEN did_purchase=1 THEN 1 ELSE 0 END) AS purchase_users_any,
          SUM(CASE WHEN did_purchase=1 AND did_cart=0 THEN 1 ELSE 0 END) AS purchase_users_without_cart
        FROM user_steps
        GROUP BY event_month
        ORDER BY event_month;
        """
    ).fetchdf()
    tracking_gap["share_without_cart"] = (
        tracking_gap["purchase_users_without_cart"] / tracking_gap["purchase_users_any"]
    )
    return funnel, tracking_gap


def query_retention(con: duckdb.DuckDBPyConnection, max_weeks: int = 8) -> pd.DataFrame:
    retention = con.execute(
        f"""
        WITH first_purchase AS (
          SELECT
            user_id,
            DATE_TRUNC('week', MIN(event_time)) AS cohort_week
          FROM events_enriched
          WHERE event_type='purchase'
          GROUP BY user_id
        ),
        purchases AS (
          SELECT
            user_id,
            DATE_TRUNC('week', event_time) AS purchase_week
          FROM events_enriched
          WHERE event_type='purchase'
        ),
        base AS (
          SELECT
            f.cohort_week,
            DATE_DIFF('week', f.cohort_week, p.purchase_week) AS week_number,
            COUNT(DISTINCT p.user_id) AS users
          FROM first_purchase f
          JOIN purchases p USING (user_id)
          GROUP BY f.cohort_week, week_number
        ),
        cohort_size AS (
          SELECT cohort_week, users AS cohort_users
          FROM base
          WHERE week_number = 0
        )
        SELECT
          b.cohort_week,
          b.week_number,
          b.users,
          c.cohort_users,
          b.users * 1.0 / c.cohort_users AS retention_rate
        FROM base b
        JOIN cohort_size c USING (cohort_week)
        WHERE b.week_number BETWEEN 0 AND {max_weeks}
        ORDER BY b.cohort_week, b.week_number;
        """
    ).fetchdf()
    retention["cohort_week"] = pd.to_datetime(retention["cohort_week"])
    return retention


def query_revenue_and_ltv(con: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    revenue = con.execute(
        """
        SELECT
          event_month,
          CASE
            WHEN price < 20 THEN 'low'
            WHEN price <= 100 THEN 'mid'
            ELSE 'high'
          END AS price_segment,
          SUM(price) AS revenue
        FROM events_enriched
        WHERE event_type = 'purchase' AND price IS NOT NULL
        GROUP BY event_month, price_segment
        ORDER BY event_month, price_segment;
        """
    ).fetchdf()

    ltv = con.execute(
        """
        WITH first_purchase AS (
          SELECT
            user_id,
            DATE_TRUNC('week', MIN(event_time)) AS cohort_week
          FROM events_enriched
          WHERE event_type='purchase'
          GROUP BY user_id
        ),
        user_revenue AS (
          SELECT
            user_id,
            SUM(price) AS user_revenue
          FROM events_enriched
          WHERE event_type='purchase' AND price IS NOT NULL
          GROUP BY user_id
        )
        SELECT
          f.cohort_week,
          COUNT(*) AS cohort_users,
          SUM(COALESCE(r.user_revenue, 0)) AS total_revenue,
          SUM(COALESCE(r.user_revenue, 0)) * 1.0 / COUNT(*) AS observed_ltv
        FROM first_purchase f
        LEFT JOIN user_revenue r USING(user_id)
        GROUP BY f.cohort_week
        ORDER BY f.cohort_week;
        """
    ).fetchdf()
    ltv["cohort_week"] = pd.to_datetime(ltv["cohort_week"])
    return revenue, ltv


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_million(x: float, _pos: int | None = None) -> str:
    return f"{x / 1_000_000:.1f}M"


def save_funnel_chart(funnel: pd.DataFrame, _tracking_gap: pd.DataFrame, out_path: Path, dpi: int) -> None:
    funnel = funnel.copy()
    funnel["event_month"] = funnel["event_month"].astype(str)
    rate_cols = ["view_to_cart_rate", "cart_to_purchase_rate", "view_to_purchase_rate"]

    x = np.arange(len(funnel))
    w = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=dpi)
    ax_rates, ax_volumes = axes

    ax_rates.bar(x - w, funnel["view_to_cart_rate"], w, color=BLUE, label="View -> Cart")
    ax_rates.bar(x, funnel["cart_to_purchase_rate"], w, color=SLATE, label="Cart -> Purchase")
    ax_rates.bar(x + w, funnel["view_to_purchase_rate"], w, color=GREEN, label="View -> Purchase")
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(funnel["event_month"])
    ax_rates.set_ylabel("Conversion rate")
    ax_rates.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax_rates.grid(axis="y", color=GRID, linewidth=1, alpha=0.8)
    ax_rates.spines["top"].set_visible(False)
    ax_rates.spines["right"].set_visible(False)
    ax_rates.legend(frameon=False, loc="upper right")

    for i in range(len(funnel)):
        for offset, col in zip([-w, 0, w], rate_cols):
            value = float(funnel.loc[i, col])
            ax_rates.text(
                x[i] + offset,
                value + 0.01,
                pct(value),
                ha="center",
                va="bottom",
                fontsize=9,
                color=MUTED,
            )

    ax_volumes.bar(x - w, funnel["view_users"], w, color=BLUE, label="View users")
    ax_volumes.bar(x, funnel["cart_users"], w, color=SLATE, label="Cart users")
    ax_volumes.bar(x + w, funnel["purchase_users"], w, color=GREEN, label="Purchase users")
    ax_volumes.set_xticks(x)
    ax_volumes.set_xticklabels(funnel["event_month"])
    ax_volumes.set_ylabel("Users")
    ax_volumes.yaxis.set_major_formatter(FuncFormatter(fmt_million))
    ax_volumes.grid(axis="y", color=GRID, linewidth=1, alpha=0.8)
    ax_volumes.spines["top"].set_visible(False)
    ax_volumes.spines["right"].set_visible(False)
    ax_volumes.legend(frameon=False, loc="upper right")
    vol_max = float(funnel[["view_users", "cart_users", "purchase_users"]].max().max())
    label_offset = vol_max * 0.012
    for i in range(len(funnel)):
        ax_volumes.text(
            x[i] - w,
            float(funnel.loc[i, "view_users"]) + label_offset,
            fmt_million(float(funnel.loc[i, "view_users"])),
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
        ax_volumes.text(
            x[i],
            float(funnel.loc[i, "cart_users"]) + label_offset,
            fmt_million(float(funnel.loc[i, "cart_users"])),
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
        ax_volumes.text(
            x[i] + w,
            float(funnel.loc[i, "purchase_users"]) + label_offset,
            fmt_million(float(funnel.loc[i, "purchase_users"])),
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_retention_chart(retention: pd.DataFrame, out_path: Path, dpi: int) -> None:
    pivot = (
        retention.pivot(index="cohort_week", columns="week_number", values="retention_rate")
        .sort_index()
        .round(4)
    )
    pivot.index = pivot.index.strftime("%Y-%m-%d")

    retention = retention.copy()
    retention["cohort_group"] = np.where(
        retention["cohort_week"] < pd.Timestamp("2019-11-01"),
        "Early cohorts (Oct)",
        "Recent cohorts (Nov)",
    )
    curves = (
        retention[retention["week_number"] <= 6]
        .groupby(["cohort_group", "week_number"], as_index=False)["retention_rate"]
        .mean()
    )
    avg_curve = retention.groupby("week_number", as_index=False)["retention_rate"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi, gridspec_kw={"width_ratios": [1.4, 1]})
    ax_heatmap, ax_curve = axes

    sns.heatmap(
        pivot,
        cmap="Blues",
        annot=True,
        fmt=".0%",
        linewidths=0.4,
        linecolor="#CBD5E1",
        cbar_kws={"label": "Retention"},
        ax=ax_heatmap,
    )
    ax_heatmap.set_xlabel("Weeks Since First Purchase")
    ax_heatmap.set_ylabel("Cohort Week")

    for group, color in [("Early cohorts (Oct)", BLUE), ("Recent cohorts (Nov)", ORANGE)]:
        subset = curves[curves["cohort_group"] == group]
        if subset.empty:
            continue
        ax_curve.plot(
            subset["week_number"],
            subset["retention_rate"],
            marker="o",
            linewidth=2.4,
            color=color,
            label=group,
        )

    ax_curve.plot(
        avg_curve["week_number"],
        avg_curve["retention_rate"],
        linestyle="--",
        linewidth=1.8,
        color=TEAL,
        label="All cohorts avg",
    )
    ax_curve.set_xlabel("Weeks Since First Purchase")
    ax_curve.set_ylabel("Retention rate")
    ax_curve.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax_curve.set_xticks(range(0, 7))
    ax_curve.set_ylim(0, 1.05)
    ax_curve.grid(axis="y", color=GRID, linewidth=1, alpha=0.8)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_revenue_chart(revenue: pd.DataFrame, ltv: pd.DataFrame, out_path: Path, dpi: int) -> None:
    revenue = revenue.copy()
    ltv = ltv.copy()
    revenue["event_month"] = revenue["event_month"].astype(str)
    seg_order = ["low", "mid", "high"]
    month_order = sorted(revenue["event_month"].unique().tolist())
    seg_colors = {"low": "#94A3B8", "mid": "#0EA5E9", "high": "#16A34A"}

    rev_pivot = (
        revenue.pivot(index="event_month", columns="price_segment", values="revenue")
        .reindex(index=month_order, columns=seg_order)
        .fillna(0.0)
    )
    totals = rev_pivot.sum(axis=1)

    ltv["cohort_label"] = ltv["cohort_week"].dt.strftime("%Y-%m-%d")
    ltv["cohort_group"] = np.where(ltv["cohort_week"] < pd.Timestamp("2019-11-01"), "Oct start", "Nov start")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi, gridspec_kw={"width_ratios": [1.1, 1]})
    ax_mix, ax_ltv = axes

    bottom = np.zeros(len(rev_pivot))
    x = np.arange(len(rev_pivot))
    for seg in seg_order:
        values = rev_pivot[seg].to_numpy()
        ax_mix.bar(
            x,
            values,
            bottom=bottom,
            color=seg_colors[seg],
            width=0.58,
            label=f"{seg.title()} price",
        )
        bottom += values

    ax_mix.set_xticks(x)
    ax_mix.set_xticklabels(rev_pivot.index)
    ax_mix.set_ylabel("Revenue")
    ax_mix.yaxis.set_major_formatter(FuncFormatter(fmt_million))
    ax_mix.grid(axis="y", color=GRID, linewidth=1, alpha=0.8)
    ax_mix.spines["top"].set_visible(False)
    ax_mix.spines["right"].set_visible(False)
    ax_mix.legend(frameon=False, loc="upper left")

    for i, total in enumerate(totals):
        ax_mix.text(i, total * 1.01, f"${total / 1_000_000:.1f}M", ha="center", va="bottom", fontsize=10, color=MUTED)

    bubble_colors = {"Oct start": BLUE, "Nov start": ORANGE}
    for group, color in bubble_colors.items():
        subset = ltv[ltv["cohort_group"] == group]
        if subset.empty:
            continue
        ax_ltv.scatter(
            subset["cohort_users"],
            subset["observed_ltv"],
            s=np.clip(subset["total_revenue"] / 300_000, 80, 700),
            alpha=0.8,
            color=color,
            label=group,
            edgecolor="white",
            linewidth=0.8,
        )

    ax_ltv.set_xlabel("Cohort users")
    ax_ltv.set_ylabel("Observed LTV (revenue per user)")
    ax_ltv.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:,.0f}"))
    ax_ltv.grid(color=GRID, linewidth=1, alpha=0.8)
    ax_ltv.spines["top"].set_visible(False)
    ax_ltv.spines["right"].set_visible(False)
    ax_ltv.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio chart assets for notebooks 02-04.")
    parser.add_argument("--output-dir", default="assets", help="Directory where PNG files will be saved.")
    parser.add_argument("--dpi", type=int, default=240, help="PNG DPI.")
    parser.add_argument("--max-weeks", type=int, default=8, help="Max retention weeks for chart generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    os.chdir(repo_root)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    con = build_connection()
    try:
        funnel, tracking_gap = query_funnel(con)
        retention = query_retention(con, max_weeks=args.max_weeks)
        revenue, ltv = query_revenue_and_ltv(con)
    finally:
        con.close()

    funnel_path = output_dir / "funnel.png"
    retention_path = output_dir / "retention.png"
    revenue_path = output_dir / "revenue.png"

    save_funnel_chart(funnel, tracking_gap, funnel_path, dpi=args.dpi)
    save_retention_chart(retention, retention_path, dpi=args.dpi)
    save_revenue_chart(revenue, ltv, revenue_path, dpi=args.dpi)

    print(f"Saved: {funnel_path}")
    print(f"Saved: {retention_path}")
    print(f"Saved: {revenue_path}")


if __name__ == "__main__":
    main()
