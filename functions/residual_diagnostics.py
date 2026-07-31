"""
Residual diagnostics for the ETF Sector Rotation project.

This is deliberately separate from portfolio.py / model evaluation. Those
answer "does the ranking signal make money out of sample" via
cross-sectional IC and a backtest. This module answers a different
question the course asks for explicitly: what does the classical
statistical structure of the model's own residuals say -- is ANY of the
unexplained variation (epsilon) noise, or is some of it structure the
model hasn't captured yet?

Tests run, on the Elastic Net (specialized) out-of-sample residuals for
target_5d_forward_return_excess_spy:

  - Durbin-Watson         : first-order autocorrelation, per ETF
  - Ljung-Box             : autocorrelation across multiple lags, per ETF,
                            at lags inside AND beyond the 5-day target
                            horizon (see note below on why both matter)
  - Breusch-Pagan / White : heteroskedasticity -- does residual variance
                            move with the fitted value, or with a known
                            volatility regime variable (VIX)?
  - Jarque-Bera           : normality / fat tails, pooled across the panel

IMPORTANT — mechanical autocorrelation from the overlapping target window:
target_5d_forward_return_excess_spy is a forward-looking 5-day return,
recomputed every trading day. Consecutive days' targets share up to 4 of
their 5 underlying return days. That overlap induces MA(4)-like
autocorrelation in the residual SERIES BY CONSTRUCTION, regardless of
whether the model is any good. So Ljung-Box rejecting at lags 1-4 is an
expected mechanical artifact, not a finding. The actual question is
whether autocorrelation PERSISTS beyond the horizon (lag > 5) -- THAT
would be structure the model has not captured, not overlap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
PREDICTIONS_PATH = MODEL_OUTPUT_DIR / "model_predictions.parquet"
MASTER_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet"

OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DATE_COL = "Date"
ETF_COL = "ETF"
TARGET_HORIZON_DAYS = 5

TARGET_COLUMN = "target_5d_forward_return_excess_spy"
MODEL = "elastic_net"
STRUCTURE = "specialized"

NAVY = "#1a2456"
GOLD = "#c9972a"


# =============================================================================
# Step 1: load residuals
# =============================================================================

def load_residuals(
    model: str = MODEL,
    structure: str = STRUCTURE,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS_PATH)
    subset = df[
        (df["target_column"] == target_column)
        & (df["model"] == model)
        & (df["structure"] == structure)
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No predictions found for {target_column}/{model}/{structure}."
        )

    subset["residual"] = subset["actual"] - subset["prediction"]
    subset = subset.sort_values([ETF_COL, DATE_COL]).reset_index(drop=True)

    # Bring in VIX as an external regime variable for the heteroskedasticity
    # test -- "does residual variance shift with a known regime", not just
    # with the model's own fitted value.
    if MASTER_DATA_PATH.exists():
        master = pd.read_parquet(MASTER_DATA_PATH, columns=[DATE_COL, "VIX"])
        master = master.drop_duplicates(DATE_COL)
        subset = subset.merge(master, on=DATE_COL, how="left")

    return subset


# =============================================================================
# Step 2: autocorrelation, per ETF (Durbin-Watson + Ljung-Box)
# =============================================================================

def autocorrelation_by_etf(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for etf, group in residuals.groupby(ETF_COL):
        group = group.sort_values(DATE_COL)
        resid = group["residual"].dropna().values

        dw = durbin_watson(resid)

        lb = acorr_ljungbox(resid, lags=[1, 4, 5, 10, 20], return_df=True)

        rows.append({
            ETF_COL: etf,
            "n_obs": len(resid),
            "durbin_watson": dw,
            "lb_pvalue_lag1": lb.loc[1, "lb_pvalue"],
            "lb_pvalue_lag4": lb.loc[4, "lb_pvalue"],
            "lb_pvalue_lag5": lb.loc[5, "lb_pvalue"],
            "lb_pvalue_lag10": lb.loc[10, "lb_pvalue"],
            "lb_pvalue_lag20": lb.loc[20, "lb_pvalue"],
        })

    result = pd.DataFrame(rows).sort_values(ETF_COL).reset_index(drop=True)

    # The interpretive flag: mechanical overlap should show up at lag <= 4.
    # Autocorrelation that SURVIVES past lag 5 is the actual "unmodeled
    # structure" candidate, not an artifact of the 5-day target window.
    result["mechanical_overlap_present"] = result["lb_pvalue_lag4"] < 0.05
    result["structure_beyond_horizon"] = (
        (result["lb_pvalue_lag10"] < 0.05) | (result["lb_pvalue_lag20"] < 0.05)
    )

    return result


# =============================================================================
# Step 3: heteroskedasticity (Breusch-Pagan + White)
# =============================================================================

def heteroskedasticity_tests(residuals: pd.DataFrame) -> dict:
    valid = residuals.dropna(subset=["residual", "prediction", "VIX"]).copy()

    resid = valid["residual"].values
    exog_bp = sm.add_constant(valid[["prediction", "VIX"]].values)

    bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(resid, exog_bp)

    exog_white = sm.add_constant(valid[["prediction"]].values)
    white_stat, white_pvalue, white_fstat, white_fpvalue = het_white(resid, exog_white)

    return {
        "n_obs": len(valid),
        "breusch_pagan_stat": bp_stat,
        "breusch_pagan_pvalue": bp_pvalue,
        "breusch_pagan_regressors": "prediction, VIX",
        "white_stat": white_stat,
        "white_pvalue": white_pvalue,
        "white_regressors": "prediction",
    }


# =============================================================================
# Step 4: normality (Jarque-Bera, pooled)
# =============================================================================

def normality_test(residuals: pd.DataFrame) -> dict:
    resid = residuals["residual"].dropna().values
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)
    return {
        "n_obs": len(resid),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "skewness": skew,
        "excess_kurtosis": kurtosis - 3,  # jarque_bera returns raw kurtosis
    }


# =============================================================================
# Step 5: plots -- residual-vs-fitted, residual-vs-time, ACF
# =============================================================================

def plot_residual_vs_fitted(residuals: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hexbin(
        residuals["prediction"], residuals["residual"],
        gridsize=45, cmap="Blues", mincnt=1,
    )
    ax.axhline(0, color=GOLD, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fitted (predicted 5-day excess-SPY return)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(
        "Residual vs. Fitted\nElastic Net (specialized), excess-SPY target",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = OUTPUT_FIGURES_DIR / "residual_vs_fitted.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_residual_vs_time(residuals: pd.DataFrame) -> Path:
    daily = residuals.groupby(DATE_COL)["residual"].mean().reset_index()
    daily = daily.sort_values(DATE_COL)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily[DATE_COL], daily["residual"], color=NAVY, linewidth=0.8)
    ax.axhline(0, color=GOLD, linewidth=1.2, linestyle="--")
    ax.set_ylabel("Mean residual across ETFs")
    ax.set_title(
        "Residual vs. Time (cross-sectional mean, by date)",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = OUTPUT_FIGURES_DIR / "residual_vs_time.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_acf_examples(residuals: pd.DataFrame, example_etfs: list[str]) -> Path:
    fig, axes = plt.subplots(1, len(example_etfs), figsize=(5 * len(example_etfs), 4.2))
    if len(example_etfs) == 1:
        axes = [axes]

    for ax, etf in zip(axes, example_etfs):
        series = residuals[residuals[ETF_COL] == etf].sort_values(DATE_COL)["residual"].dropna()
        plot_acf(series, lags=20, ax=ax, color=NAVY)
        ax.set_title(etf, fontsize=12, fontweight="bold", color=NAVY)
        ax.axvspan(0.5, 4.5, color=GOLD, alpha=0.15)  # shade the mechanical-overlap zone

    fig.suptitle(
        "Residual Autocorrelation (ACF) \u2014 shaded band = mechanical overlap zone (lags 1\u20134)",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    plt.tight_layout()
    out_path = OUTPUT_FIGURES_DIR / "residual_acf_examples.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# =============================================================================
# Model comparison: is the persistent autocorrelation an Elastic Net
# (linear-functional-form) artifact, or does it show up in LightGBM too?
#
# Logic: if a tree-based model (which can capture nonlinearities Elastic
# Net cannot) shows the SAME persistent, beyond-horizon autocorrelation,
# that is evidence AGAINST "Elastic Net's linear form is misspecified" as
# the explanation, and consistent with either a genuinely omitted variable
# or real unmodeled momentum common to both models. If LightGBM's
# autocorrelation is materially weaker, that instead points to functional-
# form misspecification in Elastic Net specifically.
#
# This test cannot fully distinguish "omitted variable" from "real
# momentum" -- both would survive this comparison equally. It only rules
# in/out one specific candidate explanation (linear misspecification).
# =============================================================================

def compare_models_autocorrelation(
    target_column: str = TARGET_COLUMN,
    structure: str = STRUCTURE,
    models: list[str] = ("elastic_net", "lightgbm"),
) -> pd.DataFrame:
    rows = []
    for model in models:
        residuals = load_residuals(model=model, structure=structure, target_column=target_column)
        autocorr = autocorrelation_by_etf(residuals)
        rows.append({
            "model": model,
            "structure": structure,
            "mean_durbin_watson": autocorr["durbin_watson"].mean(),
            "min_durbin_watson": autocorr["durbin_watson"].min(),
            "max_durbin_watson": autocorr["durbin_watson"].max(),
            "n_etfs_structure_beyond_horizon": autocorr["structure_beyond_horizon"].sum(),
            "n_etfs_total": len(autocorr),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Run everything
# =============================================================================

def main():
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = load_residuals()

    autocorr = autocorrelation_by_etf(residuals)
    autocorr.to_csv(OUTPUT_TABLES_DIR / "residual_autocorrelation_by_etf.csv", index=False)

    het = heteroskedasticity_tests(residuals)
    pd.DataFrame([het]).to_csv(OUTPUT_TABLES_DIR / "residual_heteroskedasticity.csv", index=False)

    norm = normality_test(residuals)
    pd.DataFrame([norm]).to_csv(OUTPUT_TABLES_DIR / "residual_normality.csv", index=False)

    plot_residual_vs_fitted(residuals)
    plot_residual_vs_time(residuals)
    # Pick 3 example ETFs for the ACF panel: highest, median, lowest DW
    sorted_by_dw = autocorr.sort_values("durbin_watson")
    examples = [
        sorted_by_dw.iloc[0][ETF_COL],
        sorted_by_dw.iloc[len(sorted_by_dw) // 2][ETF_COL],
        sorted_by_dw.iloc[-1][ETF_COL],
    ]
    plot_acf_examples(residuals, examples)

    print("=== Autocorrelation by ETF (Durbin-Watson + Ljung-Box) ===")
    print(autocorr.to_string(index=False))
    print(f"\nMechanical overlap (lag<=4) flagged for: "
          f"{autocorr.loc[autocorr['mechanical_overlap_present'], ETF_COL].tolist()}")
    print(f"Structure BEYOND horizon (lag 10/20) flagged for: "
          f"{autocorr.loc[autocorr['structure_beyond_horizon'], ETF_COL].tolist()}")

    print("\n=== Heteroskedasticity (Breusch-Pagan + White) ===")
    print(pd.DataFrame([het]).to_string(index=False))

    print("\n=== Normality (Jarque-Bera) ===")
    print(pd.DataFrame([norm]).to_string(index=False))

    model_comparison = compare_models_autocorrelation()
    model_comparison.to_csv(OUTPUT_TABLES_DIR / "residual_autocorrelation_model_comparison.csv", index=False)
    print("\n=== Elastic Net vs. LightGBM: does the autocorrelation persist in both? ===")
    print(model_comparison.to_string(index=False))

    return {
        "autocorrelation": autocorr,
        "heteroskedasticity": het,
        "normality": norm,
        "model_comparison": model_comparison,
    }


if __name__ == "__main__":
    main()
