
"""
Modeling pipeline for the ETF Sector Rotation project (Day 3 / Day 4).
 
Pipeline stages, in the order this file runs them:
    1. Load the feature dataset and document the feature set (internal vs.
       external, tagged with a Day 3 risk-model bucket).
    2. Standardize features point-in-time-safely (cross-sectional z-score
       for internal features, causal expanding z-score for external/macro
       features).
    3. Build embargoed walk-forward (expanding-window) CV splits.
    4. Fit two baselines (no-skill, naive momentum) as a reference point.
    5. Step 1 — feature clustering: flag redundant internal features via
       their pairwise correlation, and define a reduced feature set.
    6. Fit the two required models (Elastic Net, LightGBM), POOLED across
       all 11 ETFs, on both the full and reduced feature sets.
    7. Step 2 — pooled vs. specialized: refit both models one-ETF-at-a-time
       and compare against the pooled result.
    8. Evaluate everything with the same metric suite: time-series IC
       (primary, per Day 1/3's small-N guidance), hit rate corrected for
       the target's own base rate, and cross-sectional IC (secondary).
 
Modeling decisions, documented rather than hidden:
 
    POOLED vs. SPECIALIZED — pooling the 11 ETFs into one training set is
    a real assumption, not a free default: they have genuinely different
    macro sensitivities. Both are fit and compared (Step 2) rather than
    picking one without evidence.
 
    SMALL-N IC CAVEAT — cross-sectional Rank IC is computed across only 11
    instruments per date, which is far noisier than the same statistic
    across hundreds of names. Time-series IC (per ETF, across ~1700+
    dates) is reported as the PRIMARY metric instead; cross-sectional IC
    is kept as a secondary diagnostic, with its coverage reported
    explicitly since it can become mathematically undefined when a model
    predicts near-identical scores for every ETF on a date.
 
    HIT RATE BASE RATE — a hit rate is only informative relative to the
    rate a trivial "always predict the majority sign" rule would already
    achieve. The model uses sector-relative forward returns, whose daily
    cross-sectional mean is approximately zero, so sign accuracy measures
    relative outperformance rather than broad market direction.

    LIGHTGBM: FIXED TREE COUNT, NOT EARLY STOPPING — validation-based early
    stopping (a held-out tail slice of the training dates) was tried first
    and abandoned: across the 5 walk-forward folds, best_iteration_ landed
    at {4, 1, 67, 1, 1} — 4 of 5 folds effectively trained a single tree
    before giving up, because the validation slice is small and the
    per-round change in loss is dominated by noise rather than genuine
    over/underfitting. A fixed tree count (n_estimators=300, matched to
    the num_leaves/min_child_samples regularization already in place) is
    far more reproducible across folds, at the cost of not adapting tree
    count to how much data each fold happens to have.
"""
 
from dataclasses import dataclass
from pathlib import Path
 
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
 
 
# =============================================================================
# Project folders
# =============================================================================
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "feature_dataset.parquet"
 
 
# =============================================================================
# Config
# =============================================================================
 
# Must match features.py's add_target_variable() horizon.
TARGET_HORIZON_DAYS = 5
 
# The target at date t uses the close price at t + TARGET_HORIZON_DAYS, so
# any training row within that many days of a split boundary has a label
# window overlapping the other side of the boundary. The embargo drops
# those rows from the training side of every split — the main walk-forward
# splits below AND the inner CV folds used to tune Elastic Net.
EMBARGO_DAYS = TARGET_HORIZON_DAYS
 
ENTITY_COL = "ETF"
DATE_COL = "Date"
TARGET_COL = f"target_{TARGET_HORIZON_DAYS}d_relative_return"
 
# Full feature set, tagged (scope, Day 3 risk-model bucket) — the Day 3
# feature-documentation deliverable.
FEATURE_GROUPS = {
    "return_5d": ("internal", "fundamental (momentum)"),
    "return_20d": ("internal", "fundamental (momentum)"),
    "return_60d": ("internal", "fundamental (momentum)"),
    "price_to_MA20": ("internal", "technical (trend)"),
    "price_to_MA50": ("internal", "technical (trend)"),
    "MA20_minus_MA50": ("internal", "technical (trend)"),
    "RSI_14": ("internal", "technical (momentum oscillator)"),
    "MACD": ("internal", "technical (momentum oscillator)"),
    "volatility_20d": ("internal", "statistical (risk)"),
    "max_drawdown_60d": ("internal", "statistical (risk)"),
    "volume_change": ("internal", "statistical (trading activity)"),
    "relative_volume": ("internal", "statistical (trading activity)"),
    "VIX": ("external", "macro (sentiment / vol)"),
    "Treasury_10Y": ("external", "macro (rates)"),
    "Yield_Spread_10Y2Y": ("external", "macro (rates)"),
    "Federal_Funds_Rate": ("external", "macro (rates)"),
    "CPI": ("external", "macro (cycle)"),
    "Unemployment_Rate": ("external", "macro (cycle)"),
}
 
INTERNAL_FEATURE_COLS = [
    name for name, (scope, _) in FEATURE_GROUPS.items() if scope == "internal"
]
EXTERNAL_FEATURE_COLS = [
    name for name, (scope, _) in FEATURE_GROUPS.items() if scope == "external"
]
 
# Step 1 payoff (see compute_feature_correlation_matrix / the Step 1 block
# in __main__): 5 of the 12 internal features form one heavily redundant
# "trend/momentum-derivative" cluster (pairwise Spearman corr 0.72-0.90) —
# price_to_MA20, price_to_MA50, MA20_minus_MA50, RSI_14, MACD are all
# smoothed re-readings of the momentum signal already in the three raw
# return_Xd features. Dropping them keeps one clean representative per
# horizon instead of 8 near-duplicates. volatility_20d/max_drawdown_60d
# (corr -0.63) and volume_change/relative_volume (corr 0.50) are
# correlated but not redundant enough to collapse further.
REDUCED_INTERNAL_FEATURE_COLS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "volume_change",
    "relative_volume",
]
 
 
# =============================================================================
# Feature standardization
# =============================================================================
 
def add_cross_sectional_zscores(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Standardize internal/asset-specific features within each date's own
    cross-section (across the 11 ETFs that day) — "is this ETF extreme
    relative to its peers today," and a scale-free way to compare features
    like MACD that are naturally larger for a higher-priced ETF.
    """
 
    df = df.copy()
    grouped = df.groupby(date_col)[cols]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
 
    for col in cols:
        df[f"z_{col}"] = (df[col] - mean[col]) / std[col]
 
    return df
 
 
def add_expanding_zscores(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = DATE_COL,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    Standardize external/macro features using only PAST history up to each
    date (expanding window) — every ETF shares the same macro value on a
    given date, so a cross-sectional z-score would divide by zero. Causal
    by construction: the z-score on date t only ever uses rows with
    date <= t.
    """
 
    df = df.copy()
    daily = (
        df.drop_duplicates(subset=date_col)[[date_col] + cols]
        .sort_values(date_col)
        .set_index(date_col)
    )
 
    expanding_mean = daily.expanding(min_periods=min_periods).mean()
    expanding_std = daily.expanding(min_periods=min_periods).std().replace(0, np.nan)
 
    z = (daily - expanding_mean) / expanding_std
    z.columns = [f"z_{c}" for c in cols]
    z = z.reset_index()
 
    return df.merge(z, on=date_col, how="left")
 
 
# =============================================================================
# Step 1: feature clustering / multicollinearity diagnostics
# =============================================================================
 
def compute_feature_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Pairwise correlation across the pooled panel — Spearman to match
    the rank-based IC metrics used elsewhere."""
 
    return df[cols].corr(method=method)
 
 
def print_high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> None:
    """Flag feature pairs redundant enough to be a multicollinearity
    concern for the linear model."""
 
    cols = corr.columns.tolist()
    print(f"\nFeature pairs with |Spearman corr| > {threshold}:")
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.loc[a, b]
            if abs(c) > threshold:
                print(f"  {a:20s} {b:20s} {c:+.3f}")
 
 
# =============================================================================
# Walk-forward split generation
# =============================================================================
 
@dataclass
class WalkForwardSplit:
    split_id: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
 
 
def generate_walk_forward_splits(
    dates: pd.Series,
    n_splits: int,
    min_train_periods: int,
    embargo_days: int = EMBARGO_DAYS,
) -> list[WalkForwardSplit]:
    """
    Expanding-window walk-forward CV with an embargo gap. Fold 0 trains on
    the earliest `min_train_periods` trading days and tests on the block
    right after (minus the embargo); each later fold expands the training
    window forward in time — never shuffled, never using a date the model
    wouldn't have known about yet.
    """
 
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n_dates = len(unique_dates)
 
    if n_dates <= min_train_periods:
        raise ValueError("Not enough unique trading dates for one walk-forward split.")
 
    test_block_size = (n_dates - min_train_periods) // n_splits
    if test_block_size <= embargo_days:
        raise ValueError("Test block too small relative to the embargo.")
 
    splits = []
    for split_id in range(n_splits):
        test_start_idx = min_train_periods + split_id * test_block_size
        is_last_split = split_id == n_splits - 1
        test_end_idx = n_dates if is_last_split else test_start_idx + test_block_size
 
        test_dates = unique_dates[test_start_idx:test_end_idx]
        train_end_idx = test_start_idx - embargo_days
 
        if train_end_idx <= 0:
            raise ValueError(f"Split {split_id}: embargo consumes the entire training window.")
 
        train_dates = unique_dates[:train_end_idx]
        splits.append(WalkForwardSplit(split_id, train_dates, test_dates))
 
    return splits
 
 
def apply_split(
    df: pd.DataFrame,
    split: WalkForwardSplit,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a long-format (Date, ETF, ...) panel into train/test frames."""
 
    train_df = df[df[date_col].isin(split.train_dates)].copy()
    test_df = df[df[date_col].isin(split.test_dates)].copy()
    return train_df, test_df
 
 
def check_no_leakage(splits: list[WalkForwardSplit]) -> None:
    """Assert no date appears in both the train and test side of any split."""
 
    for split in splits:
        overlap = set(split.train_dates) & set(split.test_dates)
        if overlap:
            raise AssertionError(
                f"Split {split.split_id} has {len(overlap)} overlapping date(s) — embargo failed."
            )
 
 
# =============================================================================
# Baselines (no model fitting required)
# =============================================================================
 
def no_skill_baseline(df: pd.DataFrame, random_state: int = 42) -> pd.Series:
    """Random score per row — the floor a real model needs to clear."""
 
    rng = np.random.default_rng(random_state)
    return pd.Series(rng.standard_normal(len(df)), index=df.index)
 
 
def naive_momentum_baseline(df: pd.DataFrame, feature_col: str = "return_20d") -> pd.Series:
    """Rank ETFs by an existing momentum feature directly — no fitting.
    The check for whether the modeling complexity below is earning its
    place."""
 
    return df[feature_col]
 
 
# =============================================================================
# Evaluation metrics
# =============================================================================
 
def compute_cross_sectional_rank_ic(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    date_col: str = DATE_COL,
) -> pd.Series:
    """Spearman rank IC computed within each date's cross-section (across
    the 11 ETFs that day). SECONDARY metric — see module docstring."""
 
    def _daily_ic(group: pd.DataFrame) -> float:
        if group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            return np.nan
        return group[pred_col].corr(group[target_col], method="spearman")
 
    ic_series = df.groupby(date_col)[[pred_col, target_col]].apply(_daily_ic)
    ic_series.name = "rank_ic"
    return ic_series.dropna()
 
 
def compute_time_series_ic(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    entity_col: str = ENTITY_COL,
) -> pd.Series:
    """Spearman IC computed along TIME, separately per ETF — PRIMARY
    metric. Each correlation uses ~n_dates observations instead of just
    11, far less noisy given this project's small (N=11) universe."""
 
    def _entity_ic(group: pd.DataFrame) -> float:
        if group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            return np.nan
        return group[pred_col].corr(group[target_col], method="spearman")
 
    ts_ic = df.groupby(entity_col)[[pred_col, target_col]].apply(_entity_ic)
    ts_ic.name = "time_series_ic"
    return ts_ic.dropna()
 
 
def compute_ic_ir(ic_series: pd.Series) -> dict:
    """Descriptive mean/std/IC-IR summary without assuming independence."""

    clean = pd.Series(ic_series).dropna().astype(float)
    n = len(clean)
    mean_ic = clean.mean() if n else np.nan
    std_ic = clean.std(ddof=1) if n > 1 else np.nan
    ic_ir = mean_ic / std_ic if n > 1 and std_ic > 0 else np.nan
    return {"n_periods": n, "mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir}


def compute_hac_mean_tstat(
    ic_series: pd.Series,
    max_lag: int = TARGET_HORIZON_DAYS - 1,
) -> float:
    """Newey-West/HAC t-stat for the mean of a serially correlated IC series."""

    x = pd.Series(ic_series).dropna().astype(float).to_numpy()
    n = len(x)
    if n < 3:
        return np.nan

    demeaned = x - x.mean()
    long_run_var = float(np.dot(demeaned, demeaned) / n)
    lag_cap = min(max_lag, n - 1)

    for lag in range(1, lag_cap + 1):
        weight = 1.0 - lag / (lag_cap + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma

    if long_run_var <= 0:
        return np.nan

    return float(x.mean() / np.sqrt(long_run_var / n))


def compute_hit_rate(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    """Share of predictions with the correct sign, pooled across rows."""
 
    valid = df[[pred_col, target_col]].dropna()
    correct_sign = np.sign(valid[pred_col]) == np.sign(valid[target_col])
    return correct_sign.mean()
 
 
def compute_hit_rate_by_entity(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    entity_col: str = ENTITY_COL,
) -> pd.Series:
    """Hit rate computed separately per ETF over time."""
 
    rates = df.groupby(entity_col)[[pred_col, target_col]].apply(
        lambda g: compute_hit_rate(g, pred_col, target_col)
    )
    rates.name = "hit_rate"
    return rates
 
 
def compute_trivial_baseline_by_entity(
    df: pd.DataFrame,
    target_col: str,
    entity_col: str = ENTITY_COL,
) -> pd.Series:
    """
    The hit rate a trivial constant-sign prediction (always the majority
    sign) would already achieve, per ETF. Hit rate is only a meaningful
    skill measure relative to THIS, not to 0.5 — if the target's sign is
    skewed (e.g. an ETF that rises more often than it falls from ordinary
    market drift), a model that always predicts "up" scores a hit rate
    equal to that base rate with zero real skill.
    """
 
    rates = df.groupby(entity_col)[[target_col]].apply(
        lambda g: max((g[target_col] > 0).mean(), (g[target_col] <= 0).mean())
    )
    rates.name = "trivial_baseline_hit_rate"
    return rates
 
 
def evaluate_predictions(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
    entity_col: str = ENTITY_COL,
) -> dict:
    """
    Full evaluation suite for a small (N=11) cross-section:
      - PRIMARY: per-ETF time-series Spearman correlation, summarized across ETFs
      - PRIMARY: hit rate, and its lift over the trivial base-rate baseline
      - SECONDARY: cross-sectional rank IC, with explicit coverage
    """
 
    valid = df.dropna(subset=[pred_col, target_col]).copy()
 
    ts_ic_by_etf = compute_time_series_ic(valid, pred_col, target_col, entity_col)
    ts_ic_stats = compute_ic_ir(ts_ic_by_etf)
    ts_cross_etf_t = (
        ts_ic_stats["mean_ic"] / (ts_ic_stats["std_ic"] / np.sqrt(ts_ic_stats["n_periods"]))
        if ts_ic_stats["n_periods"] > 1 and ts_ic_stats["std_ic"] > 0
        else np.nan
    )
 
    hit_rate_by_etf = compute_hit_rate_by_entity(valid, pred_col, target_col, entity_col)
    trivial_by_etf = compute_trivial_baseline_by_entity(valid, target_col, entity_col)
    hit_rate_lift_by_etf = (hit_rate_by_etf - trivial_by_etf).rename("hit_rate_lift")
 
    cs_ic_series = compute_cross_sectional_rank_ic(valid, pred_col, target_col, date_col)
    cs_ic_stats = compute_ic_ir(cs_ic_series)
    cs_ic_stats["hac_t_stat"] = compute_hac_mean_tstat(
        cs_ic_series, max_lag=TARGET_HORIZON_DAYS - 1
    )
    cs_valid_dates = cs_ic_stats["n_periods"]
    cs_total_dates = valid[date_col].nunique()
 
    return {
        "n_obs": len(valid),
        "time_series_ic": ts_ic_stats,
        "time_series_ic_by_etf": ts_ic_by_etf,
        "time_series_cross_etf_t_stat": ts_cross_etf_t,
        "hit_rate_mean": hit_rate_by_etf.mean(),
        "hit_rate_by_etf": hit_rate_by_etf,
        "trivial_baseline_hit_rate_mean": trivial_by_etf.mean(),
        "hit_rate_lift_mean": hit_rate_lift_by_etf.mean(),
        "hit_rate_lift_by_etf": hit_rate_lift_by_etf,
        "cross_sectional_ic": cs_ic_stats,
        "cross_sectional_ic_total_dates": cs_total_dates,
        "cross_sectional_ic_coverage": f"{cs_valid_dates}/{cs_total_dates} dates",
    }
 
 
def print_evaluation_summary(name: str, results: dict) -> None:
    """Time-series metrics first (primary, N=11 universe); cross-sectional
    IC reported second as a secondary diagnostic."""
 
    ts = results["time_series_ic"]
    cs = results["cross_sectional_ic"]
 
    print(f"\n--- {name} ---")
    print(f"n_obs (out-of-sample rows): {results['n_obs']}")
    print(
        "[PRIMARY] Mean per-ETF time-series Spearman correlation: "
        f"mean={ts['mean_ic']:.4f}, std_across_ETFs={ts['std_ic']:.4f}, "
        f"cross-ETF ratio={ts['ic_ir']:.4f} (n={ts['n_periods']} ETFs)"
    )
    print(
        "          Descriptive cross-ETF t-stat "
        "(not a time-series significance test): "
        f"{results['time_series_cross_etf_t_stat']:.2f}"
    )
    print(
        f"[PRIMARY] Hit rate: {results['hit_rate_mean']:.4f}  vs. trivial "
        f"always-majority-sign baseline: {results['trivial_baseline_hit_rate_mean']:.4f}  "
        f"-> lift = {results['hit_rate_lift_mean']:+.4f}"
    )
    print(
        "[secondary] Cross-sectional rank IC: "
        f"mean={cs['mean_ic']:.4f}, HAC t-stat={cs['hac_t_stat']:.2f} "
        f"(valid on {results['cross_sectional_ic_coverage']}"
        f"{' — degenerate on the remaining dates' if cs['n_periods'] < results['cross_sectional_ic_total_dates'] else ''})"
    )
 
 
# =============================================================================
# Inner model-selection splits
# =============================================================================


def build_date_grouped_inner_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = EMBARGO_DAYS,
    date_col: str = DATE_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create inner CV indices by unique trading date, never by panel row."""

    unique_dates = pd.DatetimeIndex(sorted(pd.unique(train_df[date_col])))
    if len(unique_dates) <= n_splits + embargo_days + 1:
        raise ValueError("Not enough unique dates for date-grouped inner CV.")

    date_splitter = TimeSeriesSplit(n_splits=n_splits, gap=embargo_days)
    row_dates = pd.to_datetime(train_df[date_col]).reset_index(drop=True)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for train_date_idx, valid_date_idx in date_splitter.split(unique_dates):
        inner_train_dates = unique_dates[train_date_idx]
        inner_valid_dates = unique_dates[valid_date_idx]
        train_rows = np.flatnonzero(row_dates.isin(inner_train_dates).to_numpy())
        valid_rows = np.flatnonzero(row_dates.isin(inner_valid_dates).to_numpy())
        folds.append((train_rows, valid_rows))

    return folds


# =============================================================================
# Model 1: Penalized regression (Elastic Net) — pooled and specialized
# =============================================================================
 
def fit_predict_elastic_net(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    entity_col: str,
    target_col: str,
) -> tuple[np.ndarray, pd.Series, float, float]:
    """
    Pooled Elastic Net across all 11 ETFs, with an ETF fixed effect
    (dummy variables, one category dropped) so the model can capture a
    per-sector level shift on top of the shared feature coefficients.
 
    Inner hyperparameter search uses embargoed TimeSeriesSplit (nested CV):
    this is the INNER loop that only picks alpha/l1_ratio, nested inside
    the OUTER walk-forward split that estimates unbiased performance.
    The gap equals the target horizon to reduce overlap between training
    labels and validation dates.
    """
 
    train_df = train_df.sort_values(DATE_COL)
 
    train_dummies = pd.get_dummies(train_df[entity_col], prefix="ETF", drop_first=True).astype(float)
    test_dummies = pd.get_dummies(test_df[entity_col], prefix="ETF", drop_first=True).astype(float)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0.0)
 
    X_train = pd.concat(
        [train_df[feature_cols].reset_index(drop=True), train_dummies.reset_index(drop=True)], axis=1
    )
    X_test = pd.concat(
        [test_df[feature_cols].reset_index(drop=True), test_dummies.reset_index(drop=True)], axis=1
    )
    y_train = train_df[target_col].reset_index(drop=True)
 
    # Build the INNER CV folds from unique trading dates, not panel rows.
    inner_cv = build_date_grouped_inner_cv(
        train_df, n_splits=5, embargo_days=EMBARGO_DAYS
    )

    model = ElasticNetCV(
        # Balanced Elastic-Net-to-Lasso grid; no near-Ridge 0.01 setting.
        l1_ratio=[0.20, 0.50, 0.80, 0.95, 1.00],
        alphas=100,
        eps=1e-3,
        cv=inner_cv,
        max_iter=50_000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
 
    coefs = pd.Series(model.coef_, index=X_train.columns)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
 
    return preds, coefs, model.alpha_, model.l1_ratio_
 
 
def fit_predict_elastic_net_single(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Same Elastic Net, minus the ETF dummies — only one entity in this
    fit, used for the Step 2 specialized (per-ETF) comparison."""
 
    train_df = train_df.sort_values(DATE_COL)
    X_train = train_df[feature_cols].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)
 
    inner_cv = build_date_grouped_inner_cv(
        train_df, n_splits=5, embargo_days=EMBARGO_DAYS
    )

    model = ElasticNetCV(
        # Same non-Ridge-like grid as the pooled model.
        l1_ratio=[0.20, 0.50, 0.80, 0.95, 1.00],
        alphas=100,
        eps=1e-3,
        cv=inner_cv,
        max_iter=50_000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
 
 
# =============================================================================
# Model 2: Gradient boosting (LightGBM) — pooled and specialized
# =============================================================================
 
def fit_predict_lightgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    entity_col: str,
    target_col: str,
    n_estimators: int = 300,
) -> tuple[np.ndarray, pd.Series]:
    """
    Pooled LightGBM, fit for a FIXED number of trees — no early stopping.

    Early stopping was tried first (validation-based, on a held-out tail
    slice of the training dates) and turned out to be badly unstable at
    this data scale: across the 5 walk-forward folds, best_iteration_
    landed at {4, 1, 67, 1, 1} — i.e. 4 of 5 folds effectively trained a
    single tree before giving up, because the validation slice is small
    and the per-round change in L2 loss is dominated by noise rather than
    genuine improvement/overfitting. That instability made the reported
    metric an artifact of which fold's validation slice happened to look
    stable, not a genuine read on the model. A fixed, modest tree count
    (matched to the num_leaves/min_child_samples regularization already
    in place) is more reproducible across folds, at the cost of not
    adapting tree count to each fold's data volume.
    """

    train_X = train_df[feature_cols + [entity_col]].copy()
    test_X = test_df[feature_cols + [entity_col]].copy()

    train_X[entity_col] = train_X[entity_col].astype("category")
    test_X[entity_col] = pd.Categorical(test_X[entity_col], categories=train_X[entity_col].cat.categories)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_X, train_df[target_col], categorical_feature=[entity_col])
    preds = model.predict(test_X)

    importance = pd.Series(
        model.feature_importances_, index=train_X.columns
    ).sort_values(ascending=False)
    importance.attrs["n_estimators"] = n_estimators
    return preds, importance


def fit_predict_lightgbm_single(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_estimators: int = 300,
) -> np.ndarray:
    """Same fixed-tree-count LightGBM as fit_predict_lightgbm, minus the
    ETF categorical — used for the Step 2 specialized (per-ETF) comparison."""

    train_X = train_df[feature_cols].copy()
    test_X = test_df[feature_cols].copy()

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_X, train_df[target_col])
    return model.predict(test_X)


# =============================================================================
# Fold-by-fold runners
# =============================================================================
 
def run_walk_forward_models(
    df: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_cols: list[str],
    entity_col: str = ENTITY_COL,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fit both POOLED models fold-by-fold and collect out-of-sample
    predictions. Never re-uses a later fold's data to train an earlier
    one."""
 
    test_frames, en_snapshots, lgb_snapshots = [], {}, {}
 
    for split in splits:
        train_df, test_df = apply_split(df, split)
        train_df = train_df.dropna(subset=feature_cols + [target_col])
        test_df = test_df.dropna(subset=feature_cols + [target_col]).copy()
 
        if train_df.empty or test_df.empty:
            continue
 
        en_preds, en_coefs, alpha, l1_ratio = fit_predict_elastic_net(
            train_df, test_df, feature_cols, entity_col, target_col
        )
        lgb_preds, lgb_importance = fit_predict_lightgbm(
            train_df, test_df, feature_cols, entity_col, target_col
        )
 
        test_df["pred_elastic_net"] = en_preds
        test_df["pred_lightgbm"] = lgb_preds
        test_frames.append(test_df)
 
        en_snapshots[split.split_id] = {"alpha": alpha, "l1_ratio": l1_ratio, "coefs": en_coefs}
        lgb_snapshots[split.split_id] = lgb_importance
 
    oos_df = pd.concat(test_frames, ignore_index=True)
    return oos_df, en_snapshots, lgb_snapshots
 
 
def run_walk_forward_models_specialized(
    df: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_cols: list[str],
    entity_col: str = ENTITY_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Step 2 (Day 3): the 11 sector ETFs have genuinely different macro
    sensitivities — pooling is a modeling CHOICE, not the only valid
    default. Fits one independent Elastic Net + LightGBM per ETF, trained
    only on that ETF's own history, on the same walk-forward folds as the
    pooled version, so the two are directly comparable.
    """
 
    test_frames = []
 
    for etf in sorted(df[entity_col].unique()):
        etf_df = df[df[entity_col] == etf]
 
        for split in splits:
            train_df, test_df = apply_split(etf_df, split)
            train_df = train_df.dropna(subset=feature_cols + [target_col])
            test_df = test_df.dropna(subset=feature_cols + [target_col]).copy()
 
            if len(train_df) < 100 or test_df.empty:
                continue
 
            en_preds = fit_predict_elastic_net_single(train_df, test_df, feature_cols, target_col)
            lgb_preds = fit_predict_lightgbm_single(train_df, test_df, feature_cols, target_col)
 
            test_df["pred_elastic_net_spec"] = en_preds
            test_df["pred_lightgbm_spec"] = lgb_preds
            test_frames.append(test_df)
 
    return pd.concat(test_frames, ignore_index=True)
 
 
# =============================================================================
# Run the modeling pipeline
# =============================================================================
 
if __name__ == "__main__":
 
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {FEATURE_DATA_PATH}. Run features.py first.")
 
    feature_df = pd.read_parquet(FEATURE_DATA_PATH)
    print(f"Loaded feature dataset: {feature_df.shape}")
 
    print("\nFeature set (scope, Day 3 risk-model bucket):")
    for feature_name, (scope, bucket) in FEATURE_GROUPS.items():
        print(f"  {feature_name:22s} {scope:10s} {bucket}")
 
    # -------------------------------------------------------------------
    # Walk-forward splits
    # -------------------------------------------------------------------
 
    splits = generate_walk_forward_splits(
        dates=feature_df[DATE_COL], n_splits=5, min_train_periods=500, embargo_days=EMBARGO_DAYS
    )
 
    print(f"\nGenerated {len(splits)} walk-forward splits (embargo = {EMBARGO_DAYS} trading days):\n")
 
    test_frames = []
    for split in splits:
        train_df, test_df = apply_split(feature_df, split)
        test_frames.append(test_df)
        print(
            f"Split {split.split_id}: train {split.train_dates[0].date()} -> {split.train_dates[-1].date()} "
            f"({len(split.train_dates)} days)  |  embargo ({EMBARGO_DAYS}d)  |  "
            f"test {split.test_dates[0].date()} -> {split.test_dates[-1].date()} ({len(split.test_dates)} days)"
        )
 
    check_no_leakage(splits)
    print("\nLeakage check passed: no train/test date overlap in any split.")
 
    # -------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------
 
    baseline_oos_df = pd.concat(test_frames, ignore_index=True)
    baseline_oos_df["pred_no_skill"] = no_skill_baseline(baseline_oos_df)
    baseline_oos_df["pred_naive_momentum"] = naive_momentum_baseline(baseline_oos_df, feature_col="return_20d")
 
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (out-of-sample, all folds combined)")
    print("=" * 70)
    print_evaluation_summary("No-skill (random) baseline", evaluate_predictions(baseline_oos_df, "pred_no_skill"))
    print_evaluation_summary(
        "Naive momentum baseline (rank by return_20d)",
        evaluate_predictions(baseline_oos_df, "pred_naive_momentum"),
    )
 
    # -------------------------------------------------------------------
    # Step 1: feature clustering diagnostics
    # -------------------------------------------------------------------
 
    print("\n" + "=" * 70)
    print("STEP 1 — FEATURE CLUSTERING / MULTICOLLINEARITY DIAGNOSTICS")
    print("=" * 70)
 
    internal_corr = compute_feature_correlation_matrix(feature_df, INTERNAL_FEATURE_COLS)
    print_high_correlation_pairs(internal_corr, threshold=0.7)
    dropped = sorted(set(INTERNAL_FEATURE_COLS) - set(REDUCED_INTERNAL_FEATURE_COLS))
    print(
        f"\nDecision: reduce {len(INTERNAL_FEATURE_COLS)} internal features -> "
        f"{len(REDUCED_INTERNAL_FEATURE_COLS)} (dropped: {dropped})"
    )
 
    # -------------------------------------------------------------------
    # Standardize once (point-in-time safe), then build both feature sets
    # -------------------------------------------------------------------
 
    feature_df = add_cross_sectional_zscores(feature_df, INTERNAL_FEATURE_COLS)
    feature_df = add_expanding_zscores(feature_df, EXTERNAL_FEATURE_COLS)
 
    full_feature_cols = [f"z_{c}" for c in INTERNAL_FEATURE_COLS] + [f"z_{c}" for c in EXTERNAL_FEATURE_COLS]
    reduced_feature_cols = (
        [f"z_{c}" for c in REDUCED_INTERNAL_FEATURE_COLS] + [f"z_{c}" for c in EXTERNAL_FEATURE_COLS]
    )
 
    # -------------------------------------------------------------------
    # Pooled models: full feature set vs. reduced (Step 1) feature set
    # -------------------------------------------------------------------
 
    model_oos_df, en_snapshots, lgb_snapshots = run_walk_forward_models(feature_df, splits, full_feature_cols)
 
    print("\n" + "=" * 70)
    print("MODEL EVALUATION — FULL feature set (12 internal + 6 external)")
    print("=" * 70)
    print_evaluation_summary("Elastic Net (full)", evaluate_predictions(model_oos_df, "pred_elastic_net"))
    print_evaluation_summary("LightGBM (full)", evaluate_predictions(model_oos_df, "pred_lightgbm"))
 
    reduced_oos_df, en_snapshots_r, lgb_snapshots_r = run_walk_forward_models(
        feature_df, splits, reduced_feature_cols
    )
 
    print("\n" + "=" * 70)
    print("MODEL EVALUATION — REDUCED feature set (7 internal + 6 external)")
    print("=" * 70)
    lgb_results_r = evaluate_predictions(reduced_oos_df, "pred_lightgbm")
    print_evaluation_summary("Elastic Net (reduced)", evaluate_predictions(reduced_oos_df, "pred_elastic_net"))
    print_evaluation_summary("LightGBM (reduced)", lgb_results_r)
 
    # -------------------------------------------------------------------
    # Step 2: pooled vs. specialized (per-ETF), on the reduced feature set
    # -------------------------------------------------------------------
 
    print("\n" + "=" * 70)
    print("STEP 2 — POOLED vs. SPECIALIZED (per-ETF) MODELS")
    print("=" * 70)
 
    specialized_oos_df = run_walk_forward_models_specialized(feature_df, splits, reduced_feature_cols)
    lgb_results_spec = evaluate_predictions(specialized_oos_df, "pred_lightgbm_spec")
 
    print_evaluation_summary(
        "Elastic Net (specialized, per-ETF)", evaluate_predictions(specialized_oos_df, "pred_elastic_net_spec")
    )
    print_evaluation_summary("LightGBM (specialized, per-ETF)", lgb_results_spec)
 
    print("\nPer-ETF time-series IC, pooled vs. specialized (LightGBM):")
    comparison = pd.concat(
        [
            lgb_results_r["time_series_ic_by_etf"].rename("pooled_ic"),
            lgb_results_spec["time_series_ic_by_etf"].rename("specialized_ic"),
        ],
        axis=1,
    )
    comparison["specialized_minus_pooled"] = comparison["specialized_ic"] - comparison["pooled_ic"]
    print(comparison.sort_values("specialized_minus_pooled", ascending=False).round(4))
 
    # -------------------------------------------------------------------
    # Feature diagnostics — most recent fold, reduced set
    # -------------------------------------------------------------------
 
    last_split_id = splits[-1].split_id
 
    print("\n" + "=" * 70)
    print(f"FEATURE DIAGNOSTICS — most recent fold (split {last_split_id}), reduced set")
    print("=" * 70)
 
    en_last = en_snapshots_r[last_split_id]
    print(f"\nElastic Net: alpha={en_last['alpha']:.5f}, l1_ratio={en_last['l1_ratio']}")
    n_zeroed = (en_last["coefs"].abs() < 1e-8).sum()
    print(f"Features zeroed out by the L1 penalty: {n_zeroed} / {len(en_last['coefs'])}")
    print("Top 10 |coefficient|:")
    print(en_last["coefs"].head(10))
 
    print("\nLightGBM gain-based feature importance (top 10):")
    print(lgb_snapshots_r[last_split_id].head(10))