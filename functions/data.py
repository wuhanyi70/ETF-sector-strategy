"""
Data sourcing and cleaning pipeline for the ETF Sector Rotation project.

Tasks:
1. Download 11 sector ETF price histories.
2. Download VIX data.
3. Download selected FRED macroeconomic indicators.
4. Save raw datasets as CSV files.
5. Clean, align, and merge the datasets.
"""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred


# =============================================================================
# Project folders
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Project settings
# =============================================================================

ETF_TICKERS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

FRED_SERIES = {
    "FEDFUNDS": "Federal_Funds_Rate",
    "DGS10": "Treasury_10Y",
    "T10Y2Y": "Yield_Spread_10Y2Y",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment_Rate",
}

START_DATE = "2018-06-01"
END_DATE = "2026-07-01"

# XLC launched in June 2018.
# The balanced analysis period begins after all 11 ETFs are available.
ANALYSIS_START_DATE = "2018-07-02"

# =============================================================================
# Raw-data download functions
# =============================================================================

def download_etf_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily OHLCV data for the sector ETFs."""

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    if data.empty:
        raise RuntimeError("ETF download returned no data.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError(
            "ETF data does not have the expected multi-index columns."
        )

    downloaded_tickers = set(data.columns.get_level_values(0))
    missing_tickers = set(tickers) - downloaded_tickers

    if missing_tickers:
        raise RuntimeError(
            "ETF download is missing the following tickers: "
            f"{sorted(missing_tickers)}"
        )

    return data


def download_vix_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily VIX data from Yahoo Finance."""

    data = yf.download(
        tickers="^VIX",
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=True,
    )

    if data.empty:
        raise RuntimeError("VIX download returned no data.")

    return data


def download_fred_data(
    api_key: str,
    series_dict: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download selected macroeconomic series from FRED."""

    fred = Fred(api_key=api_key)
    downloaded_series = []

    for series_id, column_name in series_dict.items():
        series = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )

        if series.empty:
            raise RuntimeError(
                f"FRED series {series_id} returned no data."
            )

        series = series.rename(column_name)
        downloaded_series.append(series)

    macro_data = pd.concat(downloaded_series, axis=1)
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data.index.name = "Date"
    macro_data = macro_data.sort_index()

    if macro_data.empty:
        raise RuntimeError("FRED download returned no data.")

    return macro_data


# =============================================================================
# Data-cleaning and merging function
# =============================================================================

def create_master_dataset(
    etf_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    tickers: list[str],
    analysis_start_date: str,
) -> pd.DataFrame:
    """
    Convert ETF data to long format and merge VIX and macroeconomic data.

    Each output row represents one ETF on one trading date.
    """

    # -------------------------------------------------------------------------
    # 1. Convert ETF data from wide format to long format
    # -------------------------------------------------------------------------

    etf_frames = []

    for ticker in tickers:
        ticker_data = etf_data[ticker].copy()
        ticker_data["ETF"] = ticker
        ticker_data = ticker_data.reset_index()
        etf_frames.append(ticker_data)

    etf_long = pd.concat(etf_frames, ignore_index=True)

    etf_long = etf_long.rename(
        columns={
            "Adj Close": "Adj_Close",
        }
    )

    etf_long["Date"] = pd.to_datetime(etf_long["Date"])

    required_etf_columns = {
        "Date",
        "ETF",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    }

    missing_columns = required_etf_columns - set(etf_long.columns)

    if missing_columns:
        raise RuntimeError(
            "ETF data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # 2. Prepare and merge VIX closing values
    # -------------------------------------------------------------------------

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data["Close"].copy()

        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
    else:
        vix_close = vix_data["Close"].copy()

    vix_clean = vix_close.rename("VIX").reset_index()
    vix_clean["Date"] = pd.to_datetime(vix_clean["Date"])

    master_data = etf_long.merge(
        vix_clean,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------------------
    # 3. Prepare macroeconomic data
    # -------------------------------------------------------------------------

    macro_clean = macro_data.copy()
    macro_clean.index = pd.to_datetime(macro_clean.index)
    macro_clean = macro_clean.sort_index()

    monthly_columns = [
        "Federal_Funds_Rate",
        "CPI",
        "Unemployment_Rate",
    ]

    daily_columns = [
        "Treasury_10Y",
        "Yield_Spread_10Y2Y",
    ]

    required_macro_columns = set(monthly_columns + daily_columns)
    missing_macro_columns = required_macro_columns - set(macro_clean.columns)

    if missing_macro_columns:
        raise RuntimeError(
            "Macro data is missing required columns: "
            f"{sorted(missing_macro_columns)}"
        )

    monthly_macro = (
        macro_clean[monthly_columns]
        .dropna(how="all")
        .copy()
    )

    # Apply a simplified one-month availability lag.
    # Each monthly observation becomes available at the beginning
    # of the following month.
    monthly_macro.index = (
        monthly_macro.index.to_period("M").to_timestamp("M")
        + pd.Timedelta(days=1)
    )

    daily_macro = (
        macro_clean[daily_columns]
        .dropna(how="all")
        .copy()
    )

    macro_clean = pd.concat(
        [daily_macro, monthly_macro],
        axis=1,
    ).sort_index()

    # Carry each latest available macro value forward.
    macro_clean = macro_clean.ffill()

    macro_clean.index.name = "Date"
    macro_clean = macro_clean.reset_index()

    # -------------------------------------------------------------------------
    # 4. Match each trading date with the latest macro observations
    # -------------------------------------------------------------------------

    master_data = master_data.sort_values("Date")
    macro_clean = macro_clean.sort_values("Date")

    master_data = pd.merge_asof(
        master_data,
        macro_clean,
        on="Date",
        direction="backward",
    )

    # -------------------------------------------------------------------------
    # 5. Keep the balanced period when all 11 ETFs are available
    # -------------------------------------------------------------------------

    master_data = master_data[
        master_data["Date"] >= pd.Timestamp(analysis_start_date)
    ].copy()

    master_data = master_data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
        ]
    )

    master_data = master_data.sort_values(
        ["Date", "ETF"]
    ).reset_index(drop=True)

    if master_data.empty:
        raise RuntimeError(
            "The cleaned master dataset contains no observations."
        )

    final_tickers = set(master_data["ETF"].unique())
    missing_final_tickers = set(tickers) - final_tickers

    if missing_final_tickers:
        raise RuntimeError(
            "The final dataset is missing the following ETFs: "
            f"{sorted(missing_final_tickers)}"
        )

    return master_data


# =============================================================================
# Run the sourcing and cleaning pipeline
# =============================================================================

if __name__ == "__main__":

    etf_data = download_etf_data(
        tickers=ETF_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    etf_output_path = RAW_DATA_DIR / "sector_etf_prices.csv"
    etf_data.to_csv(etf_output_path)

    print(f"Saved raw ETF data to: {etf_output_path}")
    print(f"Raw ETF dataset shape: {etf_data.shape}")

    vix_data = download_vix_data(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    vix_output_path = RAW_DATA_DIR / "vix.csv"
    vix_data.to_csv(vix_output_path)

    print(f"Saved raw VIX data to: {vix_output_path}")
    print(f"Raw VIX dataset shape: {vix_data.shape}")

    fred_api_key = os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set in the terminal environment."
        )

    macro_data = download_fred_data(
        api_key=fred_api_key,
        series_dict=FRED_SERIES,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    macro_output_path = RAW_DATA_DIR / "macro_data.csv"
    macro_data.to_csv(macro_output_path)

    print(f"Saved raw FRED data to: {macro_output_path}")
    print(f"Raw FRED dataset shape: {macro_data.shape}")

    master_data = create_master_dataset(
        etf_data=etf_data,
        vix_data=vix_data,
        macro_data=macro_data,
        tickers=ETF_TICKERS,
        analysis_start_date=ANALYSIS_START_DATE,
    )

    print("\nMaster dataset validation:")
    print(f"Shape: {master_data.shape}")
    print(
        f"Date range: {master_data['Date'].min().date()} "
        f"to {master_data['Date'].max().date()}"
    )
    print(f"Number of ETFs: {master_data['ETF'].nunique()}")

    print("\nFirst 15 observations:")
    print(master_data.head(15))

    print("\nMissing values by column:")
    print(master_data.isna().sum())