"""
Figure generation for the Results & Interpretation section.

Reads the CSVs saved by portfolio.py (outputs/tables/) and produces the
presentation-ready figures used in the slides:

1. equity_curve_drawdown.png  - cumulative return + drawdown, gross vs net,
                                 with negative-Sharpe folds shaded
2. fold_sharpe_bar.png        - Sharpe ratio by walk-forward fold (the
                                 "noise vs structure" diagnostic)
3. turnover_vs_return.png     - per-period turnover against per-period
                                 net return, to make the cost-drag point
                                 visually rather than just as a number

Run after portfolio.py. All figures are saved to outputs/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Palette matches the course deck (navy / gold)
NAVY = "#1a2456"
GOLD = "#c9972a"
RED_SHADE = "#e8b4b4"
GREEN_BAR = "#3f6b4f"
RED_BAR = "#a94442"


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def plot_equity_curve_drawdown():
    bt = pd.read_csv(TABLES_DIR / "portfolio_backtest.csv", parse_dates=["Date"])
    fold_perf = pd.read_csv(
        TABLES_DIR / "portfolio_performance_by_fold.csv",
        parse_dates=["start_date", "end_date"],
    )

    bt = bt.sort_values("Date").reset_index(drop=True)
    bt["gross_cum"] = (1 + bt["gross_return"]).cumprod()
    bt["net_cum"] = (1 + bt["net_return"]).cumprod()
    bt["gross_dd"] = bt["gross_cum"] / bt["gross_cum"].cummax() - 1
    bt["net_dd"] = bt["net_cum"] / bt["net_cum"].cummax() - 1

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    ax = axes[0]
    ax.plot(bt["Date"], bt["gross_cum"], color=NAVY, linewidth=2, label="Gross")
    ax.plot(bt["Date"], bt["net_cum"], color=GOLD, linewidth=2, label="Net of cost")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

    for _, row in fold_perf.iterrows():
        if row["gross_sharpe"] < 0:
            ax.axvspan(row["start_date"], row["end_date"], color=RED_SHADE, alpha=0.35, zorder=0)

    ax.set_ylabel("Cumulative growth of $1")
    ax.set_title(
        "Long-Short Sector Rotation: Cumulative Return\n(Elastic Net, specialized, excess-SPY target)",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    ax.legend(loc="upper left", frameon=False)
    _clean_axes(ax)

    ax2 = axes[1]
    ax2.fill_between(bt["Date"], bt["gross_dd"] * 100, 0, color=NAVY, alpha=0.5, label="Gross drawdown")
    ax2.fill_between(bt["Date"], bt["net_dd"] * 100, 0, color=GOLD, alpha=0.5, label="Net drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", frameon=False, fontsize=8)
    _clean_axes(ax2)

    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = FIGURES_DIR / "equity_curve_drawdown.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_fold_sharpe_bar():
    """The noise-vs-structure diagnostic: Sharpe by walk-forward fold."""
    fold_perf = pd.read_csv(TABLES_DIR / "portfolio_performance_by_fold.csv")

    labels = [
        f"Fold {row.split_id}\n{pd.Timestamp(row.start_date):%b %Y}\u2013{pd.Timestamp(row.end_date):%b %Y}"
        for row in fold_perf.itertuples()
    ]
    colors = [GREEN_BAR if v >= 0 else RED_BAR for v in fold_perf["gross_sharpe"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, fold_perf["gross_sharpe"], color=colors, width=0.6)

    for bar, val in zip(bars, fold_perf["gross_sharpe"]):
        ax.annotate(
            f"{val:+.2f}",
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 6 if val >= 0 else -14),
            ha="center", fontsize=10, fontweight="bold", color=NAVY,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Gross Sharpe ratio")
    ax.set_title(
        "Sharpe Ratio by Walk-Forward Fold: Where the Signal Actually Worked",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    _clean_axes(ax)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fold_sharpe_bar.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_turnover_vs_return():
    """Makes the cost-drag point visually: high turnover eats the edge."""
    bt = pd.read_csv(TABLES_DIR / "portfolio_backtest.csv", parse_dates=["Date"])

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(bt["Date"], bt["turnover"] * 100, color=NAVY, alpha=0.35, width=4, label="Turnover (%)")
    ax1.set_ylabel("Turnover per rebalance (%)", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)

    ax2 = ax1.twinx()
    ax2.plot(bt["Date"], bt["net_return"].cumsum() * 100, color=GOLD, linewidth=2, label="Cumulative net return (%)")
    ax2.set_ylabel("Cumulative net return (%)", color=GOLD)
    ax2.tick_params(axis="y", labelcolor=GOLD)

    ax1.set_title(
        "Turnover Stayed High Throughout \u2014 Cost Is a Persistent Drag, Not a One-Time Hit",
        fontsize=11, fontweight="bold", color=NAVY,
    )
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = FIGURES_DIR / "turnover_vs_return.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_equity_curve_drawdown(),
        plot_fold_sharpe_bar(),
        plot_turnover_vs_return(),
    ]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
