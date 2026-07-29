"""
Feature engineering pipeline for the ETF Sector Rotation project.

Tasks:
1. Load the cleaned master dataset.
2. Compute momentum, trend, risk, trading-activity, and technical features.
3. Compute the target variable (5-day forward return).
4. Save the feature dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project folders
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_DATA_PATH = PROCESSED_DATA_DIR / "master_dataset.parquet"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "feature_dataset.parquet"


# =============================================================================
# Individual feature-group functions
# =============================================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """5-day, 20-day, and 60-day historical returns."""

    df = df.copy()
    grouped_close = df.groupby("ETF")["Adj_Close"]

    df["return_5d"] = grouped_close.pct_change(5)
    df["return_20d"] = grouped_close.pct_change(20)
    df["return_60d"] = grouped_close.pct_change(60)

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages and price-to-moving-average ratios."""

    df = df.copy()
    grouped_close = df.groupby("ETF")["Adj_Close"]

    df["MA20"] = grouped_close.transform(lambda x: x.rolling(20).mean())
    df["MA50"] = grouped_close.transform(lambda x: x.rolling(50).mean())

    df["price_to_MA20"] = df["Adj_Close"] / df["MA20"]
    df["price_to_MA50"] = df["Adj_Close"] / df["MA50"]
    df["MA20_minus_MA50"] = df["MA20"] - df["MA50"]

    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily return volatility and maximum drawdown."""

    df = df.copy()

    df["daily_return"] = df.groupby("ETF")["Adj_Close"].pct_change()

    df["volatility_20d"] = (
        df.groupby("ETF")["daily_return"]
        .transform(lambda x: x.rolling(20).std())
    )

    def _max_drawdown(prices: pd.Series, window: int = 60) -> pd.Series:
        rolling_max = prices.rolling(window).max()
        drawdown = prices / rolling_max - 1
        return drawdown.rolling(window).min()

    df["max_drawdown_60d"] = (
        df.groupby("ETF")["Adj_Close"].transform(_max_drawdown)
    )

    return df


def add_trading_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume change and relative volume."""

    df = df.copy()
    grouped_volume = df.groupby("ETF")["Volume"]

    df["volume_change"] = grouped_volume.pct_change()

    volume_ma20 = grouped_volume.transform(lambda x: x.rolling(20).mean())
    df["relative_volume"] = df["Volume"] / volume_ma20

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI (14-day) and MACD, computed manually (no external ta-lib dependency)."""

    df = df.copy()

    def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line

    df["RSI_14"] = df.groupby("ETF")["Adj_Close"].transform(_rsi)
    df["MACD"] = df.groupby("ETF")["Adj_Close"].transform(_macd)

    return df


def add_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    5-day forward return (the value the model will predict).

    This is the ONLY place where future information is intentionally used,
    since it defines the prediction target rather than a feature.
    """

    df = df.copy()

    df[f"target_{horizon}d_forward_return"] = (
        df.groupby("ETF")["Adj_Close"]
        .transform(lambda x: x.shift(-horizon) / x - 1)
    )

    return df


# =============================================================================
# Main feature-creation pipeline
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on the master dataset."""

    df = df.sort_values(["ETF", "Date"]).reset_index(drop=True)

    df = add_momentum_features(df)
    df = add_trend_features(df)
    df = add_risk_features(df)
    df = add_trading_activity_features(df)
    df = add_technical_indicators(df)
    df = add_target_variable(df, horizon=5)

    df = df.sort_values(["Date", "ETF"]).reset_index(drop=True)

    return df


# =============================================================================
# Run the feature engineering pipeline
# =============================================================================

if __name__ == "__main__":

    if not MASTER_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found at: {MASTER_DATA_PATH}. "
            "Run functions/data.py first."
        )

    master_df = pd.read_parquet(MASTER_DATA_PATH)

    print(f"Loaded master dataset: {master_df.shape}")

    feature_df = create_features(master_df)

    print(f"\nFeature dataset shape: {feature_df.shape}")
    print(f"\nColumns:\n{feature_df.columns.tolist()}")

    print("\nMissing values by column:")
    print(feature_df.isna().sum())

    print("\nFirst 15 rows:")
    print(feature_df.head(15))

    feature_df.to_parquet(
        FEATURE_DATA_PATH,
        index=False,
        engine="pyarrow",
    )

    print(f"\nSaved feature dataset to: {FEATURE_DATA_PATH}")