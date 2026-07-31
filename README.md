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
ETF-Sector-Strategy/
│
├── functions/
│   ├── data.py
│   ├── features.py
│   ├── feature_pca.py
│   ├── models.py
│   ├── portfolio.py
│   ├── reporting.py
│   └── residual_diagnostics.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── model_outputs/
│
├── README.md
├── FEATURE_DOCUMENTATION.md
├── requirements.txt
└── .gitignore
```
---

## Raw Data

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

## Analysis Period

Raw data begins on **2018-06-01**.

The analysis dataset begins on **2018-07-02**, after all 11 sector ETFs are available, and extends to **2026-06-30**.

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
python functions/features.py
python functions/feature_pca.py
python functions/models.py
python functions/portfolio.py
python functions/reporting.py
python functions/residual_diagnostics.py
```
---