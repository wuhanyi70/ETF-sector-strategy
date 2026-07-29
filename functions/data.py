"""
Raw data sourcing pipeline for the ETF Sector Rotation project.

Tasks:
1. Download 11 sector ETF price histories.
2. Download VIX data.
3. Download selected FRED macroeconomic indicators.
4. Save raw datasets as CSV files.
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

# Raw data begins one month before the balanced analysis period.
START_DATE = "2018-06-01"
END_DATE = "2026-07-01"

# XLC launched in June 2018.
# The balanced analysis period begins after all 11 ETFs are available.
ANALYSIS_START_DATE = "2018-07-02"

FRED_SERIES = {
    "FEDFUNDS": "Federal_Funds_Rate",
    "DGS10": "Treasury_10Y",
    "T10Y2Y": "Yield_Spread_10Y2Y",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment_Rate",
}


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
# Run raw-data sourcing
# =============================================================================

if __name__ == "__main__":

    # Download and save ETF data.
    etf_data = download_etf_data(
        tickers=ETF_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    etf_output_path = RAW_DATA_DIR / "sector_etf_prices.csv"
    etf_data.to_csv(etf_output_path)

    print(f"Saved raw ETF data to: {etf_output_path}")
    print(f"Raw ETF dataset shape: {etf_data.shape}")

    # Download and save VIX data.
    vix_data = download_vix_data(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    vix_output_path = RAW_DATA_DIR / "vix.csv"
    vix_data.to_csv(vix_output_path)

    print(f"Saved raw VIX data to: {vix_output_path}")
    print(f"Raw VIX dataset shape: {vix_data.shape}")

    # Obtain the FRED API key from the terminal environment.
    fred_api_key = os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set in the terminal environment."
        )

    # Download and save macroeconomic data.
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