"""
Portfolio construction and backtest for the ETF Sector Rotation project.

Takes the walk-forward, out-of-sample model predictions produced by
models.py and turns them into an actual, tradeable long-short book, then
evaluates it the way the course expects: Sharpe, Sortino, Calmar, max
drawdown, and turnover read TOGETHER, gross and net of a simple
transaction cost model, with an explicit note on whether Sharpe and IR
are the same number here.

Design decisions (documented, not hidden):
- Model / target: elastic_net, specialized (per-ETF), on
  target_5d_forward_return_excess_spy. This is the only model/target
  combination in model_evaluation.csv with a cross-sectional IC that
  clears a conventional significance bar in aggregate (HAC t-stat 2.26).
  See the by-fold diagnostic below for why that aggregate number is
  interpreted cautiously rather than taken at face value.
- Construction: long-short, dollar-neutral. Each rebalance date, rank
  all 11 ETFs by predicted excess-SPY return; go equal-weight long the
  top 3, equal-weight short the bottom 3 (+1/3 each leg, gross
  exposure = 200%, net exposure = 0%). This directly matches the
  Hit@3 metric already reported in model evaluation, rather than
  introducing a new, unvalidated cutoff.
- Rebalance cadence: every 5 trading days (non-overlapping), matching
  the model's 5-day forward-return horizon. This avoids the
  overlapping-window autocorrelation that daily rebalancing on a 5-day
  target would introduce, at the cost of a smaller effective sample of
  independent portfolio observations -- a real, disclosed trade-off.
- Cost model: linear cost only (spread/commission), applied to
  two-sided turnover at each rebalance. A market-impact (square-root)
  term is NOT applied here because the project does not have a
  per-name ADV series merged into this dataset; this is disclosed as a
  limitation, not silently assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
PREDICTIONS_PATH = MODEL_OUTPUT_DIR / "model_predictions.parquet"

OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DATE_COL = "Date"
ETF_COL = "ETF"

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PortfolioConfig:
    target_column: str = "target_5d_forward_return_excess_spy"
    model: str = "elastic_net"
    structure: str = "specialized"
    n_long: int = 3
    n_short: int = 3
    rebalance_every_n_days: int = 5
    linear_cost_bps: float = 5.0  # per-trade, one-way, in bps of traded notional
    annualization_periods_per_year: float = TRADING_DAYS_PER_YEAR / 5  # ~50.4

    # --- Market-impact cost model (disclosed assumptions, not calibrated) ---
    # Square-root impact law: impact (as a fraction of traded notional) =
    # impact_coefficient * daily_volatility_proxy * sqrt(participation_rate),
    # where participation_rate = dollar trade size / trailing ADV (dollars).
    # This requires an assumed book size, since the backtest itself works in
    # weight-fractions (a $1 book) rather than a real dollar AUM. We assume
    # a mid-size systematic book. This is explicitly a what-if / robustness
    # check, NOT a calibrated cost estimate -- the coefficient and AUM are
    # disclosed, adjustable assumptions.
    assumed_aum_usd: float = 50_000_000.0
    impact_coefficient: float = 0.3  # "Y" in the standard square-root law
    adv_lookback_days: int = 20


# =============================================================================
# Step 1: select the model/target slice and rebalance dates
# =============================================================================

def load_predictions(config: PortfolioConfig) -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS_PATH)

    subset = df[
        (df["target_column"] == config.target_column)
        & (df["model"] == config.model)
        & (df["structure"] == config.structure)
    ].copy()

    if subset.empty:
        raise RuntimeError(
            "No predictions found for "
            f"{config.target_column}/{config.model}/{config.structure}."
        )

    subset = subset.sort_values([DATE_COL, ETF_COL]).reset_index(drop=True)
    return subset


def select_rebalance_dates(
    predictions: pd.DataFrame,
    rebalance_every_n_days: int,
) -> list[pd.Timestamp]:
    """
    Non-overlapping rebalance dates, spaced by the target horizon.

    Since `actual` on a given date already IS the realized forward
    return over the next `rebalance_every_n_days` trading days, picking
    every Nth date gives non-overlapping portfolio holding periods --
    each trading day's return is counted in exactly one period.
    """
    all_dates = sorted(predictions[DATE_COL].unique())
    return list(all_dates[::rebalance_every_n_days])


# =============================================================================
# Step 2: construct target weights at each rebalance date
# =============================================================================

def construct_weights(
    predictions: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """
    Equal-weight, dollar-neutral long-short weights per rebalance date.

    Returns a long DataFrame: Date, ETF, weight, prediction, actual.
    """
    records = []

    for date in rebalance_dates:
        day = predictions[predictions[DATE_COL] == date].copy()
        day = day.sort_values("prediction", ascending=False)

        if len(day) < (n_long + n_short):
            # Not enough names to form both legs on this date; skip it
            # rather than silently forming a partial book.
            continue

        long_names = day.iloc[:n_long].copy()
        short_names = day.iloc[-n_short:].copy()

        long_names["weight"] = 1.0 / n_long
        short_names["weight"] = -1.0 / n_short

        # Any ETF not selected this period is explicitly zero-weight --
        # needed so turnover captures positions being closed out, not
        # just positions being opened.
        day["weight"] = 0.0
        day.loc[long_names.index, "weight"] = long_names["weight"]
        day.loc[short_names.index, "weight"] = short_names["weight"]

        records.append(day[[DATE_COL, ETF_COL, "weight", "prediction", "actual"]])

    weights = pd.concat(records, ignore_index=True)
    return weights


# =============================================================================
# Step 3: turn weights + realized returns into a period-by-period backtest
# =============================================================================

def backtest_portfolio(
    weights: pd.DataFrame,
    linear_cost_bps: float,
) -> pd.DataFrame:
    """
    One row per rebalance period: gross return, turnover, net return.
    """
    pivot = weights.pivot(index=DATE_COL, columns=ETF_COL, values="weight").fillna(0.0)
    actual_pivot = weights.pivot(index=DATE_COL, columns=ETF_COL, values="actual").fillna(0.0)

    pivot = pivot.sort_index()
    actual_pivot = actual_pivot.sort_index()

    # Gross portfolio return each period = sum(weight * realized forward return)
    gross_return = (pivot * actual_pivot).sum(axis=1)

    # Turnover = half the sum of absolute weight changes vs. the prior
    # period (standard definition -- counts a round-trip trade once).
    weight_changes = pivot.diff().abs().sum(axis=1)
    weight_changes.iloc[0] = pivot.iloc[0].abs().sum()  # cost of putting the book on
    turnover = 0.5 * weight_changes

    # Linear cost: turnover (as a fraction of book notional) * cost in bps
    cost = turnover * (linear_cost_bps / 10_000.0) * 2  # 2-sided: pay cost on both legs' trades

    net_return = gross_return - cost

    result = pd.DataFrame({
        "gross_return": gross_return,
        "turnover": turnover,
        "cost": cost,
        "net_return": net_return,
    })
    result.index.name = DATE_COL
    return result.reset_index()


# =============================================================================
# Step 3b: market-impact cost model (answers "does it survive real costs?")
# =============================================================================

def load_adv_and_vix(lookback_days: int) -> pd.DataFrame:
    """
    Trailing average-dollar-volume (ADV) per ETF, and VIX, from the master
    dataset. ADV uses only trailing data (rolling mean shifted by 1 day) so
    it is known before each rebalance date -- no lookahead.
    """
    master = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet",
        columns=[DATE_COL, ETF_COL, "Close", "Volume", "VIX"],
    )
    master = master.sort_values([ETF_COL, DATE_COL])
    master["dollar_volume"] = master["Close"] * master["Volume"]
    master["adv"] = (
        master.groupby(ETF_COL)["dollar_volume"]
        .transform(lambda s: s.rolling(lookback_days, min_periods=5).mean().shift(1))
    )
    return master[[DATE_COL, ETF_COL, "adv", "VIX"]]


def add_market_impact_cost(
    weights: pd.DataFrame,
    backtest: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    """
    Adds a square-root market-impact cost on top of the existing linear cost.

    impact (fraction of traded notional) = Y * daily_vol_proxy * sqrt(Q / ADV)
    where Q = assumed dollar trade size for this ETF this period, ADV = its
    trailing average dollar volume. Y and assumed AUM are disclosed
    assumptions (PortfolioConfig), not calibrated to any broker's actual
    execution data -- this is a robustness check, not a precise cost.
    """
    adv_vix = load_adv_and_vix(config.adv_lookback_days)

    w = weights.merge(adv_vix, on=[DATE_COL, ETF_COL], how="left")
    w = w.sort_values([ETF_COL, DATE_COL])
    w["weight_change"] = w.groupby(ETF_COL)["weight"].diff()
    w["weight_change"] = w["weight_change"].fillna(w["weight"])  # first trade

    w["trade_notional_usd"] = w["weight_change"].abs() * config.assumed_aum_usd
    w["participation_rate"] = (w["trade_notional_usd"] / w["adv"]).clip(upper=1.0)

    # Daily volatility proxy: trailing 20-day realized vol of the ETF's own
    # actual returns already carried in `weights` (the `actual` column, i.e.
    # realized 5-day excess return) -- reuse rather than reintroduce a new
    # unvalidated series. Approximate daily vol by dividing by sqrt(5).
    daily_vol_proxy = (
        w.groupby(ETF_COL)["actual"]
        .transform(lambda s: s.rolling(20, min_periods=5).std().shift(1))
        / np.sqrt(5)
    )

    w["impact_frac"] = (
        config.impact_coefficient * daily_vol_proxy * np.sqrt(w["participation_rate"])
    ).fillna(0.0)

    # Dollar impact cost this ETF-period, converted back to a fraction of
    # book (AUM) so it can be subtracted from the book-level return series.
    w["impact_dollar_weighted"] = w["impact_frac"] * w["weight_change"].abs()
    impact_cost_per_period = (
        w.groupby(DATE_COL)["impact_dollar_weighted"].sum().rename("impact_cost")
    )

    result = backtest.merge(impact_cost_per_period.reset_index(), on=DATE_COL, how="left")
    result["impact_cost"] = result["impact_cost"].fillna(0.0)
    result["net_return_with_impact"] = result["net_return"] - result["impact_cost"]
    return result


# =============================================================================
# Step 3c: performance split by VIX regime (more granular than 5 calendar
# folds -- directly targets "how do you know it's regime-dependent?")
# =============================================================================

def performance_by_vix_regime(
    backtest: pd.DataFrame,
    periods_per_year: float,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Splits rebalance periods into VIX terciles (Low/Mid/High, using each
    period's OWN prevailing VIX level at the rebalance date) and computes
    Sharpe within each -- a regime definition based on a real market
    variable, not just which of 5 arbitrary calendar folds a period falls
    into. More independent buckets than 5 folds, and a substantive
    (not just temporal) definition of "regime."
    """
    master = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet",
        columns=[DATE_COL, "VIX"],
    ).drop_duplicates(DATE_COL)

    bt = backtest.merge(master, on=DATE_COL, how="left")
    bt["vix_regime"] = pd.qcut(
        bt["VIX"], q=n_regimes, labels=[f"Tercile {i+1} (VIX)" for i in range(n_regimes)]
    )

    rows = []
    for regime, group in bt.groupby("vix_regime", observed=True):
        gross = compute_metrics(group["gross_return"], periods_per_year)
        net = compute_metrics(group["net_return"], periods_per_year)
        rows.append({
            "vix_regime": regime,
            "n_periods": gross["n_periods"],
            "vix_range": f"{group['VIX'].min():.1f}\u2013{group['VIX'].max():.1f}",
            "gross_sharpe": gross["sharpe"],
            "net_sharpe": net["sharpe"],
            "gross_annualized_return": gross["annualized_return"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Step 4: performance metrics
# =============================================================================

def max_drawdown(cumulative: pd.Series) -> float:
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return drawdown.min()


def compute_metrics(
    returns: pd.Series,
    periods_per_year: float,
) -> dict:
    n_periods = len(returns)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    cumulative = (1.0 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1.0
    years = n_periods / periods_per_year
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else np.nan
    sortino = (mean_return / downside_std) * np.sqrt(periods_per_year) if downside_std and downside_std > 0 else np.nan

    mdd = max_drawdown(cumulative)
    calmar = annualized_return / abs(mdd) if mdd < 0 else np.nan

    return {
        "n_periods": n_periods,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": std_return * np.sqrt(periods_per_year),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
    }


def performance_by_fold(
    backtest: pd.DataFrame,
    predictions: pd.DataFrame,
    periods_per_year: float,
) -> pd.DataFrame:
    """Break gross/net Sharpe out by walk-forward fold (split_id)."""
    date_to_split = predictions.drop_duplicates(DATE_COL).set_index(DATE_COL)["split_id"]
    bt = backtest.copy()
    bt["split_id"] = bt[DATE_COL].map(date_to_split)

    rows = []
    for split_id, group in bt.groupby("split_id"):
        gross = compute_metrics(group["gross_return"], periods_per_year)
        net = compute_metrics(group["net_return"], periods_per_year)
        rows.append({
            "split_id": split_id,
            "start_date": group[DATE_COL].min(),
            "end_date": group[DATE_COL].max(),
            "n_periods": gross["n_periods"],
            "gross_sharpe": gross["sharpe"],
            "gross_annualized_return": gross["annualized_return"],
            "gross_max_drawdown": gross["max_drawdown"],
            "net_sharpe": net["sharpe"],
            "net_annualized_return": net["annualized_return"],
            "mean_turnover": group["turnover"].mean(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Run the full pipeline
# =============================================================================

def main(config: PortfolioConfig = None) -> dict:
    if config is None:
        config = PortfolioConfig()

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(config)
    rebalance_dates = select_rebalance_dates(predictions, config.rebalance_every_n_days)
    weights = construct_weights(predictions, rebalance_dates, config.n_long, config.n_short)
    backtest = backtest_portfolio(weights, config.linear_cost_bps)

    gross_metrics = compute_metrics(backtest["gross_return"], config.annualization_periods_per_year)
    net_metrics = compute_metrics(backtest["net_return"], config.annualization_periods_per_year)
    fold_perf = performance_by_fold(backtest, predictions, config.annualization_periods_per_year)

    # Market-impact robustness check (disclosed assumptions -- see PortfolioConfig)
    backtest = add_market_impact_cost(weights, backtest, config)
    impact_metrics = compute_metrics(backtest["net_return_with_impact"], config.annualization_periods_per_year)

    # VIX-regime split -- more independent buckets than 5 calendar folds
    regime_perf = performance_by_vix_regime(backtest, config.annualization_periods_per_year)

    backtest.to_csv(OUTPUT_TABLES_DIR / "portfolio_backtest.csv", index=False)
    fold_perf.to_csv(OUTPUT_TABLES_DIR / "portfolio_performance_by_fold.csv", index=False)
    regime_perf.to_csv(OUTPUT_TABLES_DIR / "portfolio_performance_by_vix_regime.csv", index=False)

    summary = pd.DataFrame([
        {"book": "gross", **gross_metrics, "mean_turnover": backtest["turnover"].mean()},
        {"book": "net (linear cost)", **net_metrics, "mean_turnover": backtest["turnover"].mean()},
        {"book": "net (linear + market impact)", **impact_metrics, "mean_turnover": backtest["turnover"].mean()},
    ])
    summary.to_csv(OUTPUT_TABLES_DIR / "portfolio_summary_metrics.csv", index=False)

    print("=== Portfolio summary (gross vs net vs net+impact) ===")
    print(summary.to_string(index=False))
    print("\n=== Performance by walk-forward fold ===")
    print(fold_perf.to_string(index=False))
    print("\n=== Performance by VIX regime (tercile) ===")
    print(regime_perf.to_string(index=False))
    print(
        f"\nAssumptions for market impact: AUM=${config.assumed_aum_usd:,.0f}, "
        f"impact coefficient Y={config.impact_coefficient}, "
        f"ADV lookback={config.adv_lookback_days}d. "
        "This is a disclosed what-if check, not a calibrated cost estimate."
    )

    return {
        "backtest": backtest,
        "fold_performance": fold_perf,
        "vix_regime_performance": regime_perf,
        "gross_metrics": gross_metrics,
        "net_metrics": net_metrics,
        "impact_metrics": impact_metrics,
    }


if __name__ == "__main__":
    main()
