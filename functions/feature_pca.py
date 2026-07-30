"""
PCA diagnostic test for the ETF Sector Rotation feature set.

Purpose:
This is NOT a feature-construction step. It is a diagnostic tool to
check for multicollinearity / redundancy across the engineered feature
set (cf. Day 3 slide: "Multicollinearity & Feature Clustering").

It answers: how many independent dimensions of information do our
features actually contain, and which features cluster together?

Tasks:
1. Load the feature dataset.
2. Select numeric feature columns (exclude identifiers, raw prices,
   and the target).
3. Standardize features (required before PCA — unequal scales would
   otherwise dominate the components).
4. Run PCA and report explained variance, cumulative variance, and
   the top loadings per component.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project folders
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "feature_dataset.parquet"


# =============================================================================
# Columns to exclude from the PCA test
# (identifiers, raw price levels, and the target are not "features"
# in the modeling sense — including raw prices would also dominate
# the PCA purely due to scale/trend, not genuine feature redundancy)
# =============================================================================

EXCLUDE_COLUMNS = [
    "Date",
    "ETF",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Volume",
    "SPY",
    "daily_return",
    "target_5d_forward_return",
]


# =============================================================================
# PCA diagnostic function
# =============================================================================

def run_feature_pca(
    df: pd.DataFrame,
    exclude_columns: list[str] = None,
    n_components: int = None,
) -> dict:
    """
    Run a diagnostic PCA on the engineered feature set.

    Returns a dict with:
    - 'explained_variance_ratio': variance explained by each component
    - 'cumulative_variance_ratio': running total
    - 'loadings': DataFrame of component loadings per feature
    - 'feature_columns': the columns actually used
    - 'n_obs_used': number of complete-case rows used
    """

    if exclude_columns is None:
        exclude_columns = EXCLUDE_COLUMNS

    feature_columns = [
        col for col in df.columns
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # PCA requires complete cases; rolling-window features produce
    # NaNs in the first ~60 rows per ETF, which we drop here.
    feature_data = df[feature_columns].dropna()

    if feature_data.empty:
        raise RuntimeError(
            "No complete-case rows remain after dropping NaNs. "
            "Check that the feature dataset was built correctly."
        )

    if n_components is None:
        n_components = min(len(feature_columns), feature_data.shape[0])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)

    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "feature_columns": feature_columns,
        "n_obs_used": feature_data.shape[0],
    }


def print_pca_summary(results: dict, top_n_loadings: int = 10, top_n_components: int = 12) -> None:
    """Print a readable summary of the PCA diagnostic results."""

    print(f"Number of features tested: {len(results['feature_columns'])}")
    print(f"Number of complete-case observations used: {results['n_obs_used']}")

    print("\nExplained variance by component:")
    for i in range(min(top_n_components, len(results["explained_variance_ratio"]))):
        var = results["explained_variance_ratio"][i]
        cum = results["cumulative_variance_ratio"][i]
        print(f"  PC{i + 1}: {var:.1%} (cumulative: {cum:.1%})")

    print(f"\nTop {top_n_loadings} feature loadings per component:")
    for i in range(min(top_n_components, len(results["explained_variance_ratio"]))):
        col = f"PC{i + 1}"
        top_features = results["loadings"][col].abs().sort_values(ascending=False).head(top_n_loadings)
        print(f"\n  {col}:")
        for feat, loading_abs in top_features.items():
            signed_loading = results["loadings"].loc[feat, col]
            print(f"    {feat:30s} {signed_loading:+.3f}")


# =============================================================================
# Run the diagnostic
# =============================================================================

if __name__ == "__main__":

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at: {FEATURE_DATA_PATH}. "
            "Run functions/features.py first."
        )

    feature_df = pd.read_parquet(FEATURE_DATA_PATH)
    print(f"Loaded feature dataset: {feature_df.shape}")

    results = run_feature_pca(feature_df)

    print_pca_summary(results, top_n_components=12)

    # How many components are needed to explain 90% of feature variance?
    n_for_90pct = int(np.argmax(results["cumulative_variance_ratio"] >= 0.90) + 1)
    print(
        f"\nNumber of components needed to explain 90% of feature "
        f"variance: {n_for_90pct} (out of {len(results['feature_columns'])} "
        f"total features)"
    )