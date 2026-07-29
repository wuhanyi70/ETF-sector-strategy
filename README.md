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
| Yahoo Finance | Sector ETF OHLCV prices |
| Yahoo Finance | CBOE Volatility Index (VIX) |
| FRED | Federal Funds Rate |
| FRED | 10-Year Treasury Yield |
| FRED | 10Y–2Y Yield Spread |
| FRED | Consumer Price Index (CPI) |
| FRED | Unemployment Rate |

---

## Data Pipeline

1. Download raw ETF price data.
2. Download VIX data.
3. Download macroeconomic data from FRED.
4. Save raw datasets as CSV files.
5. Merge all datasets into a single analysis-ready dataset.
6. Export the cleaned dataset as a Parquet file.

---

## Current Processing Decisions

### Universe

The strategy uses the following 11 SPDR sector ETFs:

- XLB
- XLC
- XLE
- XLF
- XLI
- XLK
- XLP
- XLRE
- XLU
- XLV
- XLY

### Analysis Period

Raw data begins on **2018-06-01**.

The analysis dataset begins on **2018-07-02**, after all 11 sector ETFs are available.

### Point-in-Time Assumptions

To reduce look-ahead bias, monthly macroeconomic variables (e.g., CPI and unemployment rate) are shifted to the beginning of the following month before being merged with daily market data. This approximates the fact that these indicators are published after the period they measure.

Daily Treasury and Federal Funds series remain on their original observation dates because they are available at daily frequency.

Before merging, macroeconomic series are forward-filled between observation dates. The macro dataset is then merged with ETF trading data using a backward `merge_asof`, ensuring that each trading day is matched only with the most recently available macroeconomic information.

---

## Reproducibility

Install dependencies

```bash
pip install -r requirements.txt
```

Set your FRED API key

```bash
export FRED_API_KEY="YOUR_API_KEY"
```

Run

```bash
python functions/data.py
```

---