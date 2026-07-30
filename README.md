# MMF1927H – Workshop in Mathematical Finance

# Sector ETF Rotation Strategy

**Team Members**
- Hanyi (Emily) Wu
- Xinyi (Cindy) Shi
- Ruoyu (Laura) Zhu
- Yue (Daisy) Yin

---

## Project Overview

The objective of this project is to predict the relative returns of the 11 SPDR Sector ETFs using market and macroeconomic information.

The project follows a standard machine learning workflow:

1. Data collection
2. Feature engineering
3. Model training
4. Portfolio construction
5. Performance evaluation

---

## Project Structure

```
ETF-Sector-Rotation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── functions/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── portfolio.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── main.ipynb
```
---

## Data Sources

| Source | Variables |
|---------|-----------|
| Yahoo Finance | 11 SPDR Sector ETF OHLCV prices |
| Yahoo Finance | SPY (S&P 500 ETF) |
| Yahoo Finance | CBOE Volatility Index (VIX) |
| FRED | Effective Federal Funds Rate |
| FRED | 2-Year Treasury Yield |
| FRED | 10-Year Treasury Yield |
| FRED | 10Y–2Y Yield Spread |
| FRED | 10-Year Breakeven Inflation Rate (T10YIE) |
| FRED | National Financial Conditions Index (NFCI) |
| FRED | Consumer Price Index (CPI) |
| FRED | Unemployment Rate |

---

## Data Pipeline

1. Download SPDR Sector ETF price data from Yahoo Finance.
2. Download SPY data.
3. Download VIX data.
4. Download macroeconomic and market data from FRED.
5. Save all raw datasets as CSV files.
6. Clean and align datasets using point-in-time assumptions.
7. Merge all datasets into a single analysis-ready dataset.
8. Export the processed dataset as a Parquet file.

---

## Current Processing Decisions

### Analysis Period

Raw data begins on **2018-06-01**.

The analysis dataset begins on **2018-07-02**, after all 11 sector ETFs are available.

### Point-in-Time Assumptions

To reduce look-ahead bias, monthly macroeconomic variables (Federal Funds Rate, Consumer Price Index, and Unemployment Rate) are shifted to the beginning of the following month before being merged with daily market data. This approximates the delay between the observation period and public availability of these indicators.

Daily market variables (2-Year Treasury Yield, 10-Year Treasury Yield, 10Y–2Y Yield Spread, and 10-Year Breakeven Inflation Rate) remain on their original observation dates because they are assumed to be observable on those dates.

The National Financial Conditions Index (NFCI) is released weekly and remains on its original observation dates. 

Before merging, all macroeconomic series are forward-filled between observation dates. The macro dataset is then merged with ETF trading data using a backward `merge_asof`, ensuring that each trading day is matched only with the most recently available information.

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your FRED API key:

```bash
export FRED_API_KEY="YOUR_API_KEY"
```

Run the data pipeline:

```bash
python functions/data.py
```

The script downloads all raw datasets, processes them, and exports the master dataset in Parquet format.

---