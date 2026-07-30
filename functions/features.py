"""
Feature engineering pipeline for the ETF Sector Rotation project.

Tasks:
1. Load the cleaned master dataset (includes SPY and expanded macro data).
2. Compute internal (endogenous) features: returns (raw / relative /
   excess), trend, risk, trading activity, technical indicators, and
   statistical factors.
3. Compute external (exogenous) features: macro exposures.
4. Compute the target variable (5-day forward return).
5. Save the feature dataset with documentation.
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

MOMENTUM_HORIZONS = [5, 20, 60]

# Trading days per year, used to convert the annualized Fed Funds Rate
# (quoted in percent) into a horizon-equivalent risk-free return.
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Feature documentation table
# (Category: Internal/External. Bucket: Fundamental/Statistical/Macro)
# =============================================================================

FEATURE_DOCUMENTATION = {
    # Raw returns
    "return_5d":                 ("Internal", "Fundamental", "Raw 5-day return"),
    "return_20d":                ("Internal", "Fundamental", "Raw 20-day (1-month) return"),
    "return_60d":                ("Internal", "Fundamental", "Raw 60-day (3-month) return"),
    # Relative returns (vs. SPY market benchmark)
    "return_5d_rel_spy":         ("Internal (derived)", "Fundamental", "5-day return relative to SPY (market benchmark)"),
    "return_20d_rel_spy":        ("Internal (derived)", "Fundamental", "20-day return relative to SPY (market benchmark)"),
    "return_60d_rel_spy":        ("Internal (derived)", "Fundamental", "60-day return relative to SPY (market benchmark)"),
    # Excess returns (vs. risk-free rate)
    "return_5d_excess_rf":       ("Internal (derived)", "Fundamental", "5-day return in excess of the risk-free rate (Fed Funds Rate)"),
    "return_20d_excess_rf":      ("Internal (derived)", "Fundamental", "20-day return in excess of the risk-free rate (Fed Funds Rate)"),
    "return_60d_excess_rf":      ("Internal (derived)", "Fundamental", "60-day return in excess of the risk-free rate (Fed Funds Rate)"),
    # Trend
    "MA20":                      ("Internal", "Fundamental", "Trend"),
    "MA50":                      ("Internal", "Fundamental", "Trend"),
    "price_to_MA20":              ("Internal", "Fundamental", "Trend strength"),
    "price_to_MA50":              ("Internal", "Fundamental", "Trend strength"),
    "MA20_minus_MA50":            ("Internal", "Fundamental", "Trend direction"),
    # Risk
    "volatility_20d":             ("Internal", "Fundamental", "Risk proxy"),
    "max_drawdown_60d":           ("Internal", "Fundamental", "Downside risk"),
    "ATR_14":                     ("Internal", "Fundamental", "Paper-derived (Gong & Mueller, 2026): volatility, highest-importance group in their RF model"),
    # Trading activity
    "volume_change":              ("Internal", "Fundamental", "Trading activity"),
    "relative_volume":            ("Internal", "Fundamental", "Trading activity"),
    # Technical
    "RSI_14":                     ("Internal", "Fundamental", "Technical / overbought-oversold"),
    "MACD":                       ("Internal", "Fundamental", "Technical / trend strength"),
    # Statistical
    "market_beta_60d":            ("Internal (derived)", "Statistical", "Rolling beta vs SPY"),
    "idio_vol_60d":               ("Internal (derived)", "Statistical", "Residual volatility after removing SPY-driven component"),
    # Macro
    "rate_sensitivity_beta":      ("External", "Macro", "Rolling beta vs change in 10Y Treasury yield"),
    "VIX":                        ("External", "Macro", "Market-wide risk/fear gauge"),
    "Financial_Conditions_Index": ("External", "Macro", "Composite financial conditions (NFCI)"),
    "Breakeven_Inflation_10Y":    ("External", "Macro", "Forward-looking inflation expectation"),
    "Federal_Funds_Rate":         ("External", "Macro", "Interest-rate environment / risk-free rate"),
    "Treasury_10Y":               ("External", "Macro", "Long-term rate level"),
    "Yield_Spread_10Y2Y":         ("External", "Macro", "Business-cycle / recession signal"),
    "CPI":                        ("External", "Macro", "Realized inflation (lagged)"),
    "Unemployment_Rate":          ("External", "Macro", "Labor market strength"),
    # Target
    "target_5d_forward_return":   ("Target", "N/A", "5-day forward return (prediction target)"),
}


# =============================================================================
# Internal / Fundamental features — returns (raw / relative / excess)
# =============================================================================

def add_return_features(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """
    For each horizon, compute three versions of the return:

    1. Raw return     — the ETF's own historical return, pct_change(h).
    2. Relative return — raw return minus SPY's return over the same
                          horizon. Answers "did this sector beat the
                          broad market?"
    3. Excess return   — raw return minus the risk-free rate compounded
                          over the same horizon. Answers "did this sector
                          beat what you'd earn holding cash?" This is the
                          classical excess-return definition used in
                          Sharpe-ratio-style analysis.

    The risk-free rate is proxied by the Fed Funds Rate (annualized,
    quoted in percent) and converted to a horizon-equivalent return
    using simple (non-compounded) scaling, which is a standard
    approximation for short holding periods.
    """

    if horizons is None:
        horizons = MOMENTUM_HORIZONS

    df = df.copy()

    # Horizon-equivalent risk-free return, derived from the annualized
    # Fed Funds Rate. E.g. a 5% annualized rate over a 20-day horizon:
    # 0.05 * (20 / 252).
    daily_rf_rate = (df["Federal_Funds_Rate"] / 100) / TRADING_DAYS_PER_YEAR

    for h in horizons:
        raw_col = f"return_{h}d"
        rel_col = f"return_{h}d_rel_spy"
        excess_col = f"return_{h}d_excess_rf"

        # 1. Raw return
        df[raw_col] = df.groupby("ETF")["Adj_Close"].pct_change(h)

        # 2. Relative return (vs. SPY)
        spy_return_col = f"_spy_return_{h}d"
        df[spy_return_col] = df.groupby("ETF")["SPY"].transform(
            lambda x, h=h: x.pct_change(h)
        )
        df[rel_col] = df[raw_col] - df[spy_return_col]
        df = df.drop(columns=[spy_return_col])

        # 3. Excess return (vs. risk-free rate, horizon-equivalent)
        horizon_rf_return = daily_rf_rate * h
        df[excess_col] = df[raw_col] - horizon_rf_return

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


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Paper-derived feature (Gong & Mueller, 2026, Journal of Empirical
    Finance): Average True Range. Their random forest feature-importance
    analysis found the volatility indicator group to be the single most
    important category (relative importance 0.36 of 5 groups), ahead of
    trend and momentum.
    """

    df = df.copy()

    def _true_range(sub_df: pd.DataFrame) -> pd.Series:
        high_low = sub_df["High"] - sub_df["Low"]
        high_close = (sub_df["High"] - sub_df["Adj_Close"].shift(1)).abs()
        low_close = (sub_df["Low"] - sub_df["Adj_Close"].shift(1)).abs()
        true_range = pd.concat(
            [high_low, high_close, low_close], axis=1
        ).max(axis=1)
        return true_range.rolling(window).mean()

    df["ATR_14"] = df.groupby("ETF", group_keys=False).apply(_true_range)

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
    """RSI (14-day) and MACD, computed manually."""

    df = df.copy()

    def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    df["RSI_14"] = df.groupby("ETF")["Adj_Close"].transform(_rsi)
    df["MACD"] = df.groupby("ETF")["Adj_Close"].transform(_macd)

    return df


# =============================================================================
# Internal / Statistical features (require SPY as market benchmark)
# =============================================================================

def add_market_beta_and_idio_vol(
    df: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    """
    Rolling market beta against SPY, and idiosyncratic volatility
    (residual volatility after removing the SPY-driven component).
    Uses SPY rather than the 11-ETF average as the market proxy, since
    the 11 sector ETFs are themselves subsets of the S&P 500.
    """

    df = df.copy()
    df["spy_daily_return"] = df.groupby("ETF")["SPY"].transform(
        lambda x: x.pct_change()
    )

    def _rolling_beta(sub_df: pd.DataFrame) -> pd.Series:
        cov = sub_df["daily_return"].rolling(window).cov(sub_df["spy_daily_return"])
        var = sub_df["spy_daily_return"].rolling(window).var()
        return cov / var

    df["market_beta_60d"] = df.groupby("ETF", group_keys=False).apply(_rolling_beta)

    df["idio_return"] = (
        df["daily_return"] - df["market_beta_60d"] * df["spy_daily_return"]
    )
    df["idio_vol_60d"] = (
        df.groupby("ETF")["idio_return"]
        .transform(lambda x: x.rolling(window).std())
    )

    df = df.drop(columns=["spy_daily_return", "idio_return"])

    return df


# =============================================================================
# External / Macro features
# =============================================================================

def add_rate_sensitivity_beta(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    External / Macro feature: rolling beta of each ETF's return against
    the change in the 10Y Treasury yield. Duration-like macro exposure.
    """

    df = df.copy()
    df["treasury_10y_change"] = df.groupby("ETF")["Treasury_10Y"].diff()

    def _rolling_beta(sub_df: pd.DataFrame) -> pd.Series:
        cov = sub_df["daily_return"].rolling(window).cov(sub_df["treasury_10y_change"])
        var = sub_df["treasury_10y_change"].rolling(window).var()
        return cov / var

    df["rate_sensitivity_beta"] = (
        df.groupby("ETF", group_keys=False).apply(_rolling_beta)
    )

    df = df.drop(columns=["treasury_10y_change"])

    return df


# =============================================================================
# Target variable
# =============================================================================

def add_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    5-day forward return — the only place future information is
    intentionally used, since this defines the prediction target
    rather than a feature.
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

    # Internal / Fundamental
    df = add_return_features(df)
    df = add_trend_features(df)
    df = add_risk_features(df)
    df = add_atr(df)
    df = add_trading_activity_features(df)
    df = add_technical_indicators(df)

    # Internal / Statistical (requires SPY / cross-sectional structure)
    df = add_market_beta_and_idio_vol(df)

    # External / Macro
    df = add_rate_sensitivity_beta(df)

    # Target
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

    print("\nFeature documentation:")
    for feature, (category, bucket, rationale) in FEATURE_DOCUMENTATION.items():
        print(f"  {feature:35s} | {category:22s} | {bucket:12s} | {rationale}")

    feature_df.to_parquet(
        FEATURE_DATA_PATH,
        index=False,
        engine="pyarrow",
    )

    print(f"\nSaved feature dataset to: {FEATURE_DATA_PATH}")