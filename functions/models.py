"""
Modeling pipeline for the ETF Sector Rotation project.

This version is aligned with the current functions/features.py and
functions/feature_pca.py, and folds in a round of code review:

- Imports SELECTED_FEATURES and TARGET_COLUMNS from feature_pca.py.
- Uses both current targets:
    * target_5d_forward_return_raw
    * target_5d_forward_return_excess_spy
- Automatically detects date-shared versus ETF-varying features from the
  constructed dataset, then applies the appropriate point-in-time scaling.
- Compares pooled and specialized Elastic Net / LightGBM models.
- Pooled models now get an ETF fixed effect (dummy variables for Elastic
  Net, a categorical feature for LightGBM). Internal features are already
  z-scored within each date's cross-section, so on their own a pooled
  model can only express "how does this ETF compare to its peers TODAY" --
  it had no way to express "this sector structurally outperforms," which
  is exactly what let historical_mean and the specialized (per-ETF)
  models beat pooled Elastic Net/LightGBM on Hit@1 in the earlier run.
- Tunes Elastic Net using true date-level cross-sectional IC (pooled) or
  time-series rank IC (specialized, where cross-sectional IC is
  undefined for a single ETF).
- Evaluates daily cross-sectional IC with a Newey-West/HAC t-stat, BOTH
  pooled across all outer folds (model_evaluation.csv) and broken out
  fold-by-fold (model_evaluation_by_fold.csv) -- a pooled-only summary
  can hide a result that is really concentrated in one or two folds.
- Reports an overall sign hit rate, Hit@1 and Hit@3, each with skill vs.
  its correct baseline -- majority-sign for the overall hit rate, and
  K / n_ETFs random selection for Hit@K, NOT 50%, which is only the
  right null for a balanced binary sign call.
- LightGBM is fit on a more lenient train/test frame that only requires
  a non-null target (LightGBM handles missing features natively), so it
  no longer loses rows to Elastic Net's complete-case requirement --
  most relevant during the walk-forward warm-up window when long-lookback
  features (e.g. the 60-day rolling features) are still NaN.
- Quiet by default: progress/diagnostic prints are gated behind
  ModelConfig.verbose (main(verbose=True) to restore full logging). The
  saved CSVs are unaffected either way.
- Saves one prediction file, one pooled-across-folds evaluation file, one
  per-fold evaluation file, and the coefficient / importance diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pca import SELECTED_FEATURES, TARGET_COLUMNS

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)
# =============================================================================
# Paths and columns
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.parquet"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"

DATE_COL = "Date"
ETF_COL = "ETF"
TARGET_HORIZON_DAYS = 5

FEATURE_COLUMNS = list(SELECTED_FEATURES)
TARGET_COLUMNS = list(TARGET_COLUMNS)

# The feature names are maintained only in feature_pca.py. The modeling
# pipeline does not duplicate them or manually label macro/ETF features.
# Instead, it inspects the actual feature dataset:
#   - date-shared feature: one value across all ETFs on a date
#   - cross-sectional feature: values vary across ETFs on at least one date
# This correctly treats rate_sensitivity_beta as cross-sectional because
# features.py estimates it separately for each ETF.
STANDARDIZED_FEATURE_COLUMNS = [f"z_{feature}" for feature in FEATURE_COLUMNS]


# =============================================================================
# Configuration and result containers
# =============================================================================

@dataclass
class ModelConfig:
    n_outer_splits: int = 5
    n_inner_splits: int = 3
    embargo_periods: int = TARGET_HORIZON_DAYS
    random_state: int = 42
    n_jobs: int = -1
    min_specialized_train_rows: int = 150
    verbose: bool = False
    # Populated in main() from the actual dataset so pooled models always
    # see the same, fixed set of ETF dummy/categorical levels regardless
    # of which subset of ETFs happens to appear in a given fold.
    entity_categories: list[str] = field(default_factory=list)


@dataclass
class WalkForwardSplit:
    split_id: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    train_index: np.ndarray
    test_index: np.ndarray


ELASTIC_NET_PARAM_GRID = {
    "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}


# =============================================================================
# Data preparation
# =============================================================================

def load_feature_dataset(file_path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {file_path}. Run functions/features.py first."
        )

    df = pd.read_parquet(file_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    return df.sort_values([DATE_COL, ETF_COL]).reset_index(drop=True)


def validate_dataset(df: pd.DataFrame, verbose: bool = False) -> None:
    required = [DATE_COL, ETF_COL] + FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The feature dataset is not aligned with the current feature_pca.py. "
            f"Missing columns: {missing}"
        )

    duplicate_rows = df.duplicated([DATE_COL, ETF_COL]).sum()
    if duplicate_rows:
        raise ValueError(
            f"Found {duplicate_rows} duplicate Date/ETF observations."
        )

    if verbose:
        print("Dataset validation passed.")
        print(f"  Selected features: {len(FEATURE_COLUMNS)}")
        print(f"  Targets: {TARGET_COLUMNS}")


def _safe_cross_sectional_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def infer_feature_structure(
    df: pd.DataFrame, verbose: bool = False
) -> tuple[list[str], list[str]]:
    """Infer preprocessing groups directly from the constructed dataset."""
    date_shared: list[str] = []
    cross_sectional: list[str] = []

    for feature in FEATURE_COLUMNS:
        # Ignore missing values when checking whether ETFs share the same
        # value on each date. A feature is date-shared only when it never has
        # more than one observed value across ETFs on any date.
        max_values_per_date = df.groupby(DATE_COL)[feature].nunique(dropna=True).max()
        if pd.isna(max_values_per_date) or max_values_per_date <= 1:
            date_shared.append(feature)
        else:
            cross_sectional.append(feature)

    if set(date_shared) | set(cross_sectional) != set(FEATURE_COLUMNS):
        raise RuntimeError("Automatic feature classification was incomplete.")

    if verbose:
        print("\nAutomatically inferred feature structure:")
        print(f"  Date-shared features ({len(date_shared)}): {date_shared}")
        print(f"  Cross-sectional features ({len(cross_sectional)}): {cross_sectional}")

    return date_shared, cross_sectional


def standardize_cross_sectional_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for feature in feature_columns:
        df[f"z_{feature}"] = (
            df.groupby(DATE_COL)[feature]
            .transform(_safe_cross_sectional_zscore)
        )
    return df


def standardize_date_shared_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if not feature_columns:
        return df.copy()

    df = df.copy()
    daily = (
        df.groupby(DATE_COL, as_index=False)[feature_columns]
        .first()
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    for feature in feature_columns:
        # shift(1) prevents the current observation from influencing its own
        # expanding mean and standard deviation.
        expanding_mean = daily[feature].expanding().mean().shift(1)
        expanding_std = daily[feature].expanding().std(ddof=0).shift(1)
        daily[f"z_{feature}"] = (
            (daily[feature] - expanding_mean) / expanding_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df.merge(
        daily[[DATE_COL] + [f"z_{f}" for f in feature_columns]],
        on=DATE_COL,
        how="left",
        validate="many_to_one",
    )


def prepare_model_dataset(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    date_shared, cross_sectional = infer_feature_structure(df, verbose=verbose)
    df = standardize_cross_sectional_features(df, cross_sectional)
    df = standardize_date_shared_features(df, date_shared)
    return df.sort_values([DATE_COL, ETF_COL]).reset_index(drop=True)


# =============================================================================
# Pooled design matrix (adds an ETF fixed effect)
# =============================================================================

def build_pooled_design_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    entity_categories: list[str],
) -> pd.DataFrame:
    """
    Standardized features plus one-hot ETF dummies (one category dropped
    as the reference level). `entity_categories` is fixed from the full
    dataset (not just this fold), so train and test always share the same
    dummy columns even if a fold happens to be missing an ETF.
    """

    entity = pd.Categorical(df[ETF_COL], categories=entity_categories)
    dummies = pd.get_dummies(entity, prefix="ETF", drop_first=True).astype(float)
    dummies.index = df.index

    return pd.concat(
        [df[feature_columns].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )


# =============================================================================
# Walk-forward splits
# =============================================================================

def generate_walk_forward_splits(
    df: pd.DataFrame,
    config: ModelConfig,
) -> list[WalkForwardSplit]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(df[DATE_COL])))
    n_dates = len(unique_dates)
    test_window_size = n_dates // (config.n_outer_splits + 1)

    if test_window_size <= config.embargo_periods:
        raise ValueError("Not enough dates for the requested splits and embargo.")

    splits: list[WalkForwardSplit] = []

    for split_number in range(config.n_outer_splits):
        test_start = test_window_size * (split_number + 1)
        test_end = (
            n_dates
            if split_number == config.n_outer_splits - 1
            else test_start + test_window_size
        )
        train_end = test_start - config.embargo_periods

        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[test_start:test_end]

        train_index = df.index[df[DATE_COL].isin(train_dates)].to_numpy()
        test_index = df.index[df[DATE_COL].isin(test_dates)].to_numpy()

        if len(train_index) == 0 or len(test_index) == 0:
            raise ValueError(f"Outer split {split_number + 1} is empty.")

        splits.append(
            WalkForwardSplit(
                split_id=split_number + 1,
                train_dates=train_dates,
                test_dates=test_dates,
                train_index=train_index,
                test_index=test_index,
            )
        )

    _check_no_split_leakage(splits)

    return splits


def _check_no_split_leakage(splits: list[WalkForwardSplit]) -> None:
    """Safety net: no outer split should ever have a date on both sides."""
    for split in splits:
        overlap = set(split.train_dates) & set(split.test_dates)
        if overlap:
            raise AssertionError(
                f"Outer split {split.split_id} has {len(overlap)} "
                "date(s) on both the train and test side."
            )


def generate_inner_time_splits(
    train_df: pd.DataFrame,
    config: ModelConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(train_df[DATE_COL])))
    validation_size = len(unique_dates) // (config.n_inner_splits + 1)

    if validation_size <= config.embargo_periods:
        raise ValueError("Not enough dates for inner CV and embargo.")

    row_dates = train_df[DATE_COL].reset_index(drop=True)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for split_number in range(config.n_inner_splits):
        validation_start = validation_size * (split_number + 1)
        validation_end = (
            len(unique_dates)
            if split_number == config.n_inner_splits - 1
            else validation_start + validation_size
        )
        training_end = validation_start - config.embargo_periods

        inner_train_dates = unique_dates[:training_end]
        inner_validation_dates = unique_dates[validation_start:validation_end]

        inner_train_rows = np.flatnonzero(
            row_dates.isin(inner_train_dates).to_numpy()
        )
        inner_validation_rows = np.flatnonzero(
            row_dates.isin(inner_validation_dates).to_numpy()
        )

        if len(inner_train_rows) == 0 or len(inner_validation_rows) == 0:
            raise ValueError("An inner CV fold is empty.")

        folds.append((inner_train_rows, inner_validation_rows))

    return folds


# =============================================================================
# Date-level cross-sectional IC tuning
# =============================================================================

def mean_daily_cross_sectional_ic(
    dates: pd.Series,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    scoring_df = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(dates).to_numpy(),
            "actual": np.asarray(actual, dtype=float),
            "prediction": np.asarray(prediction, dtype=float),
        }
    )

    daily_ics: list[float] = []
    for _, group in scoring_df.groupby(DATE_COL):
        if (
            group["prediction"].nunique() < 2
            or group["actual"].nunique() < 2
        ):
            continue

        ic, _ = spearmanr(group["prediction"], group["actual"])
        if np.isfinite(ic):
            daily_ics.append(float(ic))

    return float(np.mean(daily_ics)) if daily_ics else -1.0


def build_elastic_net_pipeline(config: ModelConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=20_000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def tune_elastic_net_by_daily_ic(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: ModelConfig,
    scoring_mode: str = "daily_cross_sectional_ic",
    include_entity_dummies: bool = False,
) -> tuple[Pipeline, dict, float, list[str]]:
    train_df = train_df.sort_values([DATE_COL, ETF_COL]).reset_index(drop=True)

    if include_entity_dummies:
        X = build_pooled_design_matrix(
            train_df, feature_columns, config.entity_categories
        )
    else:
        X = train_df[feature_columns]

    y = train_df[target_column]
    folds = generate_inner_time_splits(train_df, config)

    best_score = -np.inf
    best_params: dict | None = None

    for params in ParameterGrid(ELASTIC_NET_PARAM_GRID):
        fold_scores: list[float] = []

        for train_rows, validation_rows in folds:
            model = clone(build_elastic_net_pipeline(config))
            model.set_params(**params)
            model.fit(X.iloc[train_rows], y.iloc[train_rows])

            validation_prediction = model.predict(X.iloc[validation_rows])

            if scoring_mode == "daily_cross_sectional_ic":
                fold_score = mean_daily_cross_sectional_ic(
                    dates=train_df.iloc[validation_rows][DATE_COL],
                    actual=y.iloc[validation_rows].to_numpy(),
                    prediction=validation_prediction,
                )
            elif scoring_mode == "time_series_rank_ic":
                y_validation = y.iloc[validation_rows].to_numpy()
                if (
                    np.unique(validation_prediction).size < 2
                    or np.unique(y_validation).size < 2
                ):
                    fold_score = -1.0
                else:
                    fold_score, _ = spearmanr(
                        validation_prediction,
                        y_validation,
                    )
                    fold_score = (
                        float(fold_score)
                        if np.isfinite(fold_score)
                        else -1.0
                    )
            else:
                raise ValueError(
                    "scoring_mode must be 'daily_cross_sectional_ic' "
                    "or 'time_series_rank_ic'."
                )

            fold_scores.append(fold_score)

        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Elastic Net hyperparameter search failed.")

    final_model = build_elastic_net_pipeline(config)
    final_model.set_params(**best_params)
    final_model.fit(X, y)

    return final_model, best_params, best_score, list(X.columns)


# =============================================================================
# Model fitting
# =============================================================================

def _prediction_frame(
    test_df: pd.DataFrame,
    prediction: np.ndarray,
    target_column: str,
    split_id: int,
    model: str,
    structure: str,
) -> pd.DataFrame:
    result = test_df[[DATE_COL, ETF_COL, target_column]].copy()
    result = result.rename(columns={target_column: "actual"})
    result["prediction"] = prediction
    result["split_id"] = split_id
    result["target_column"] = target_column
    result["model"] = model
    result["structure"] = structure
    return result[
        [
            DATE_COL,
            ETF_COL,
            "split_id",
            "target_column",
            "model",
            "structure",
            "actual",
            "prediction",
        ]
    ]


def fit_pooled_elastic_net(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    split_id: int,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, best_params, best_score, design_columns = tune_elastic_net_by_daily_ic(
        train_df,
        STANDARDIZED_FEATURE_COLUMNS,
        target_column,
        config,
        include_entity_dummies=True,
    )

    X_test = build_pooled_design_matrix(
        test_df, STANDARDIZED_FEATURE_COLUMNS, config.entity_categories
    )
    prediction = model.predict(X_test)
    pred_df = _prediction_frame(
        test_df, prediction, target_column, split_id, "elastic_net", "pooled"
    )

    coefficients = pd.DataFrame(
        {
            "feature": design_columns,
            "coefficient": model.named_steps["model"].coef_,
        }
    )
    coefficients["selected"] = ~np.isclose(coefficients["coefficient"], 0.0)
    coefficients["split_id"] = split_id
    coefficients["target_column"] = target_column
    coefficients["alpha"] = best_params["model__alpha"]
    coefficients["l1_ratio"] = best_params["model__l1_ratio"]
    coefficients["inner_cv_daily_cs_ic"] = best_score

    if config.verbose:
        print(
            f"  Elastic Net pooled | alpha={best_params['model__alpha']:.6g} "
            f"| l1_ratio={best_params['model__l1_ratio']:.2f} "
            f"| inner daily CS IC={best_score:+.4f} "
            f"| nonzero={int(coefficients['selected'].sum())}/{len(coefficients)}"
        )

    return pred_df, coefficients


def fit_pooled_lightgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    split_id: int,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols_with_entity = STANDARDIZED_FEATURE_COLUMNS + [ETF_COL]
    train_X = train_df[feature_cols_with_entity].copy()
    test_X = test_df[feature_cols_with_entity].copy()

    train_X[ETF_COL] = pd.Categorical(
        train_X[ETF_COL], categories=config.entity_categories
    )
    test_X[ETF_COL] = pd.Categorical(
        test_X[ETF_COL], categories=config.entity_categories
    )

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="regression",
        importance_type="gain",
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )

    model.fit(
        train_X,
        train_df[target_column],
        categorical_feature=[ETF_COL],
    )
    prediction = model.predict(test_X)

    pred_df = _prediction_frame(
        test_df, prediction, target_column, split_id, "lightgbm", "pooled"
    )
    importance = pd.DataFrame(
        {
            "feature": train_X.columns,
            "gain_importance": model.feature_importances_,
            "split_id": split_id,
            "target_column": target_column,
        }
    )

    return pred_df, importance


def fit_specialized_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lgbm_train_df: pd.DataFrame,
    lgbm_test_df: pd.DataFrame,
    target_column: str,
    split_id: int,
    config: ModelConfig,
) -> list[pd.DataFrame]:
    """
    `train_df`/`test_df` are the complete-case frames used by Elastic Net.
    `lgbm_train_df`/`lgbm_test_df` only require a non-null target -- they
    may still contain NaN features, which LightGBM handles natively, so
    specialized LightGBM does not lose rows to Elastic Net's imputation
    requirement (most relevant during the walk-forward warm-up window).
    """

    results: list[pd.DataFrame] = []

    for etf in sorted(test_df[ETF_COL].unique()):
        etf_train = train_df[train_df[ETF_COL] == etf].copy()
        etf_test = test_df[test_df[ETF_COL] == etf].copy()

        etf_lgbm_train = lgbm_train_df[lgbm_train_df[ETF_COL] == etf].copy()
        etf_lgbm_test = lgbm_test_df[lgbm_test_df[ETF_COL] == etf].copy()

        if etf_test.empty:
            continue

        if len(etf_train) < config.min_specialized_train_rows:
            fallback = np.repeat(etf_train[target_column].mean(), len(etf_test))
            results.append(
                _prediction_frame(
                    etf_test,
                    fallback,
                    target_column,
                    split_id,
                    "historical_mean",
                    "specialized_fallback",
                )
            )
            continue

        # A single-ETF model has no cross-section within a date, so true
        # date-level cross-sectional IC is mathematically undefined here.
        # Specialized Elastic Net is therefore tuned on time-series rank IC.
        en_model, _, _, _ = tune_elastic_net_by_daily_ic(
            etf_train,
            STANDARDIZED_FEATURE_COLUMNS,
            target_column,
            config,
            scoring_mode="time_series_rank_ic",
        )
        en_prediction = en_model.predict(etf_test[STANDARDIZED_FEATURE_COLUMNS])
        results.append(
            _prediction_frame(
                etf_test,
                en_prediction,
                target_column,
                split_id,
                "elastic_net",
                "specialized",
            )
        )

        if len(etf_lgbm_train) < config.min_specialized_train_rows or etf_lgbm_test.empty:
            continue

        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="regression",
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbosity=-1,
        )
        lgb_model.fit(
            etf_lgbm_train[STANDARDIZED_FEATURE_COLUMNS],
            etf_lgbm_train[target_column],
        )
        lgb_prediction = lgb_model.predict(
            etf_lgbm_test[STANDARDIZED_FEATURE_COLUMNS]
        )
        results.append(
            _prediction_frame(
                etf_lgbm_test,
                lgb_prediction,
                target_column,
                split_id,
                "lightgbm",
                "specialized",
            )
        )

    return results


def make_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    split_id: int,
) -> list[pd.DataFrame]:
    zero_prediction = np.zeros(len(test_df))
    zero = _prediction_frame(
        test_df,
        zero_prediction,
        target_column,
        split_id,
        "zero_return",
        "baseline",
    )

    etf_means = train_df.groupby(ETF_COL)[target_column].mean()
    global_mean = train_df[target_column].mean()
    historical_prediction = (
        test_df[ETF_COL].map(etf_means).fillna(global_mean).to_numpy()
    )
    historical = _prediction_frame(
        test_df,
        historical_prediction,
        target_column,
        split_id,
        "historical_mean",
        "baseline",
    )

    return [zero, historical]


# =============================================================================
# Evaluation
# =============================================================================

def compute_hac_mean_tstat(
    series: pd.Series,
    max_lag: int = TARGET_HORIZON_DAYS - 1,
) -> float:
    x = pd.Series(series).dropna().astype(float).to_numpy()
    n = len(x)
    if n < 3:
        return np.nan

    demeaned = x - x.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    lag_cap = min(max_lag, n - 1)

    for lag in range(1, lag_cap + 1):
        weight = 1.0 - lag / (lag_cap + 1.0)
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * covariance

    if long_run_variance <= 0:
        return np.nan

    return float(x.mean() / np.sqrt(long_run_variance / n))


def daily_cross_sectional_ic(predictions: pd.DataFrame) -> pd.Series:
    def _daily_ic(group: pd.DataFrame) -> float:
        if (
            group["prediction"].nunique() < 2
            or group["actual"].nunique() < 2
        ):
            return np.nan
        ic, _ = spearmanr(group["prediction"], group["actual"])
        return float(ic) if np.isfinite(ic) else np.nan

    result = predictions.groupby(DATE_COL).apply(
        _daily_ic,
        include_groups=False,
    )
    return result.dropna()


def time_series_ic_by_etf(predictions: pd.DataFrame) -> pd.Series:
    def _etf_ic(group: pd.DataFrame) -> float:
        if (
            group["prediction"].nunique() < 2
            or group["actual"].nunique() < 2
        ):
            return np.nan
        ic, _ = spearmanr(group["prediction"], group["actual"])
        return float(ic) if np.isfinite(ic) else np.nan

    return predictions.groupby(ETF_COL).apply(
        _etf_ic,
        include_groups=False,
    ).dropna()


def directional_metrics(predictions: pd.DataFrame) -> dict:
    valid = predictions[["prediction", "actual"]].dropna().copy()

    # A zero prediction is neutral, not a directional call.
    directional = valid[~np.isclose(valid["prediction"], 0.0)].copy()
    coverage = len(directional) / len(valid) if len(valid) else np.nan

    if directional.empty:
        accuracy = np.nan
        predicted_positive_rate = np.nan
    else:
        accuracy = (
            np.sign(directional["prediction"])
            == np.sign(directional["actual"])
        ).mean()
        predicted_positive_rate = (directional["prediction"] > 0).mean()

    return {
        # Overall hit rate = sign accuracy across every observation on which
        # the model makes a non-zero directional call. This is different from
        # Hit@K, which evaluates only the top-ranked ETFs each date.
        "overall_hit_rate": accuracy,
        "directional_coverage": coverage,
        "actual_positive_rate": (
            (valid["actual"] > 0).mean() if len(valid) else np.nan
        ),
        "predicted_positive_rate": predicted_positive_rate,
        "majority_sign_baseline": (
            max((valid["actual"] > 0).mean(), (valid["actual"] < 0).mean())
            if len(valid) else np.nan
        ),
    }


def top_k_hit_rate(
    predictions: pd.DataFrame,
    k: int,
) -> float:
    daily_rates: list[float] = []

    for _, group in predictions.groupby(DATE_COL):
        if len(group) < k or group["prediction"].nunique() < 2:
            continue

        predicted_top = set(group.nlargest(k, "prediction")[ETF_COL])
        actual_top = set(group.nlargest(k, "actual")[ETF_COL])
        daily_rates.append(len(predicted_top & actual_top) / k)

    return float(np.mean(daily_rates)) if daily_rates else np.nan


def top_k_random_baseline(predictions: pd.DataFrame, k: int) -> float:
    """
    Expected Hit@K under random top-K selection: K / n_ETFs -- NOT 50%.
    50% is only the right null for a balanced binary sign call; here the
    question is "did my K picks overlap the actual top K out of n_ETFs."
    """
    n_entities = predictions[ETF_COL].nunique()
    return k / n_entities if n_entities else np.nan


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    valid = predictions.dropna(subset=["actual", "prediction"]).copy()
    error = valid["prediction"] - valid["actual"]

    cs_ic = daily_cross_sectional_ic(valid)
    ts_ic = time_series_ic_by_etf(valid)
    direction = directional_metrics(valid)

    denominator = float(
        ((valid["actual"] - valid["actual"].mean()) ** 2).sum()
    )
    r_squared = (
        1.0 - float((error ** 2).sum()) / denominator
        if denominator > 0
        else np.nan
    )

    overall_hit_rate_skill = (
        direction["overall_hit_rate"] - direction["majority_sign_baseline"]
        if np.isfinite(direction["overall_hit_rate"])
        and np.isfinite(direction["majority_sign_baseline"])
        else np.nan
    )

    hit_at_1 = top_k_hit_rate(valid, 1)
    hit_at_3 = top_k_hit_rate(valid, 3)
    hit_at_1_random = top_k_random_baseline(valid, 1)
    hit_at_3_random = top_k_random_baseline(valid, 3)

    return {
        "n_obs": len(valid),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r_squared": r_squared,
        "pearson_correlation": valid["prediction"].corr(valid["actual"]),
        "mean_time_series_ic": ts_ic.mean(),
        "n_time_series_ic_etfs": len(ts_ic),
        "mean_daily_cross_sectional_ic": cs_ic.mean(),
        "daily_cs_ic_hac_t_stat": compute_hac_mean_tstat(cs_ic),
        "positive_daily_ic_rate": (cs_ic > 0).mean() if len(cs_ic) else np.nan,
        "valid_daily_ic_dates": len(cs_ic),
        "total_test_dates": valid[DATE_COL].nunique(),
        "hit_rate_at_1": hit_at_1,
        "hit_rate_at_1_random_baseline": hit_at_1_random,
        "hit_rate_at_1_vs_random": (
            hit_at_1 - hit_at_1_random
            if np.isfinite(hit_at_1) and np.isfinite(hit_at_1_random)
            else np.nan
        ),
        "hit_rate_at_3": hit_at_3,
        "hit_rate_at_3_random_baseline": hit_at_3_random,
        "hit_rate_at_3_vs_random": (
            hit_at_3 - hit_at_3_random
            if np.isfinite(hit_at_3) and np.isfinite(hit_at_3_random)
            else np.nan
        ),
        "overall_hit_rate_skill_vs_majority": overall_hit_rate_skill,
        **direction,
    }


def evaluate_all_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled-across-all-outer-folds evaluation: one row per
    (target_column, model, structure)."""

    records: list[dict] = []

    group_columns = ["target_column", "model", "structure"]
    for keys, group in predictions.groupby(group_columns, dropna=False):
        target_column, model, structure = keys
        record = {
            "target_column": target_column,
            "model": model,
            "structure": structure,
        }
        record.update(evaluate_predictions(group))
        records.append(record)

    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


def evaluate_all_models_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Same metrics as evaluate_all_models, but one row per
    (target_column, model, structure, split_id) -- needed to check whether
    a result like a marginally-significant HAC t-stat is stable across the
    outer walk-forward folds or concentrated in just one or two of them.
    """

    records: list[dict] = []

    group_columns = ["target_column", "model", "structure", "split_id"]
    for keys, group in predictions.groupby(group_columns, dropna=False):
        target_column, model, structure, split_id = keys
        record = {
            "target_column": target_column,
            "model": model,
            "structure": structure,
            "split_id": split_id,
        }
        record.update(evaluate_predictions(group))
        records.append(record)

    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


# =============================================================================
# Main runner
# =============================================================================

def main(verbose: bool = False) -> None:
    config = ModelConfig(verbose=verbose)

    df = load_feature_dataset()
    validate_dataset(df, verbose=config.verbose)
    df = prepare_model_dataset(df, verbose=config.verbose)

    config.entity_categories = sorted(df[ETF_COL].unique().tolist())

    splits = generate_walk_forward_splits(df, config)

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for target_column in TARGET_COLUMNS:
        if config.verbose:
            print("\n" + "=" * 78)
            print(f"TARGET: {target_column}")
            print("=" * 78)

        target_df = df.dropna(subset=[target_column]).copy()

        for split in splits:
            raw_train_df = target_df.loc[
                target_df.index.intersection(split.train_index)
            ].copy()
            raw_test_df = target_df.loc[
                target_df.index.intersection(split.test_index)
            ].copy()

            # Elastic Net needs complete cases; LightGBM handles missing
            # feature values natively and gets a more lenient frame below,
            # so it does not lose rows Elastic Net has to drop.
            train_df = raw_train_df.dropna(subset=STANDARDIZED_FEATURE_COLUMNS)
            test_df = raw_test_df.dropna(subset=STANDARDIZED_FEATURE_COLUMNS)

            lgbm_train_df = raw_train_df
            lgbm_test_df = raw_test_df

            if train_df.empty or test_df.empty:
                continue

            if config.verbose:
                print(
                    f"Split {split.split_id}: "
                    f"train={train_df[DATE_COL].nunique()} dates, "
                    f"test={test_df[DATE_COL].nunique()} dates"
                )

            prediction_frames.extend(
                make_baselines(
                    train_df, test_df, target_column, split.split_id
                )
            )

            en_predictions, coefficients = fit_pooled_elastic_net(
                train_df,
                test_df,
                target_column,
                split.split_id,
                config,
            )
            prediction_frames.append(en_predictions)
            coefficient_frames.append(coefficients)

            if not lgbm_train_df.empty and not lgbm_test_df.empty:
                lgb_predictions, importance = fit_pooled_lightgbm(
                    lgbm_train_df,
                    lgbm_test_df,
                    target_column,
                    split.split_id,
                    config,
                )
                prediction_frames.append(lgb_predictions)
                importance_frames.append(importance)

            prediction_frames.extend(
                fit_specialized_models(
                    train_df,
                    test_df,
                    lgbm_train_df,
                    lgbm_test_df,
                    target_column,
                    split.split_id,
                    config,
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    evaluation = evaluate_all_models(predictions)
    evaluation_by_fold = evaluate_all_models_by_fold(predictions)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = MODEL_OUTPUT_DIR / "model_predictions.parquet"
    evaluation_path = MODEL_OUTPUT_DIR / "model_evaluation.csv"
    evaluation_by_fold_path = MODEL_OUTPUT_DIR / "model_evaluation_by_fold.csv"

    predictions.to_parquet(prediction_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    evaluation_by_fold.to_csv(evaluation_by_fold_path, index=False)

    if coefficient_frames:
        pd.concat(coefficient_frames, ignore_index=True).to_csv(
            MODEL_OUTPUT_DIR / "elastic_net_coefficients.csv",
            index=False,
        )

    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            MODEL_OUTPUT_DIR / "lightgbm_gain_importance.csv",
            index=False,
        )

    if config.verbose:
        print("\n" + "=" * 78)
        print("FINAL EVALUATION (pooled across folds)")
        print("=" * 78)
        print(evaluation.to_string(index=False))

    print(f"Saved predictions:          {prediction_path}")
    print(f"Saved evaluation (overall): {evaluation_path}")
    print(f"Saved evaluation (by fold): {evaluation_by_fold_path}")


if __name__ == "__main__":
    main()
