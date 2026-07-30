# Feature Documentation

This table is generated from `FEATURE_DOCUMENTATION` in [functions/features.py](functions/features.py).
Category: Internal/External. Bucket: Fundamental/Statistical/Macro.

| Feature | Category | Bucket | Rationale |
|---|---|---|---|
| **Raw returns** | | | |
| `return_5d` | Internal | Fundamental | Raw 5-day return |
| `return_20d` | Internal | Fundamental | Raw 20-day (1-month) return |
| `return_60d` | Internal | Fundamental | Raw 60-day (3-month) return |
| **Relative returns (vs. SPY market benchmark)** | | | |
| `return_5d_rel_spy` | Internal (derived) | Fundamental | 5-day return relative to SPY (market benchmark) |
| `return_20d_rel_spy` | Internal (derived) | Fundamental | 20-day return relative to SPY (market benchmark) |
| `return_60d_rel_spy` | Internal (derived) | Fundamental | 60-day return relative to SPY (market benchmark) |
| **Excess returns (vs. risk-free rate)** | | | |
| `return_5d_excess_rf` | Internal (derived) | Fundamental | 5-day return in excess of the risk-free rate (Fed Funds Rate) |
| `return_20d_excess_rf` | Internal (derived) | Fundamental | 20-day return in excess of the risk-free rate (Fed Funds Rate) |
| `return_60d_excess_rf` | Internal (derived) | Fundamental | 60-day return in excess of the risk-free rate (Fed Funds Rate) |
| **Trend** | | | |
| `MA20` | Internal | Fundamental | Trend |
| `MA50` | Internal | Fundamental | Trend |
| `price_to_MA20` | Internal | Fundamental | Trend strength |
| `price_to_MA50` | Internal | Fundamental | Trend strength |
| `MA20_minus_MA50` | Internal | Fundamental | Trend direction |
| **Risk** | | | |
| `volatility_20d` | Internal | Fundamental | Risk proxy |
| `max_drawdown_60d` | Internal | Fundamental | Downside risk |
| `ATR_14` | Internal | Fundamental | Paper-derived (Gong & Mueller, 2026): volatility, highest-importance group in their RF model |
| **Trading activity** | | | |
| `volume_change` | Internal | Fundamental | Trading activity |
| `relative_volume` | Internal | Fundamental | Trading activity |
| **Technical** | | | |
| `RSI_14` | Internal | Fundamental | Technical / overbought-oversold |
| `MACD` | Internal | Fundamental | Technical / trend strength |
| **Statistical** | | | |
| `market_beta_60d` | Internal (derived) | Statistical | Rolling beta vs SPY |
| `idio_vol_60d` | Internal (derived) | Statistical | Residual volatility after removing SPY-driven component |
| **Macro** | | | |
| `rate_sensitivity_beta` | External | Macro | Rolling beta vs change in 10Y Treasury yield |
| `VIX` | External | Macro | Market-wide risk/fear gauge |
| `Financial_Conditions_Index` | External | Macro | Composite financial conditions (NFCI) |
| `Breakeven_Inflation_10Y` | External | Macro | Forward-looking inflation expectation |
| `Federal_Funds_Rate` | External | Macro | Interest-rate environment / risk-free rate |
| `Treasury_10Y` | External | Macro | Long-term rate level |
| `Yield_Spread_10Y2Y` | External | Macro | Business-cycle / recession signal |
| `CPI` | External | Macro | Realized inflation (lagged) |
| `Unemployment_Rate` | External | Macro | Labor market strength |
| **Targets (two candidates — final choice still to be decided)** | | | |
| `target_5d_forward_return_raw` | Target | N/A | Raw 5-day forward return. Right default if the project's thesis is factor timing (systematic market swings ARE the signal being predicted). |
| `target_5d_forward_return_excess_spy` | Target | N/A | 5-day forward return net of SPY's same-horizon return (factor-neutral proxy). Right default if the project's thesis is idiosyncratic-alpha (sector-specific mispricing, not a market call). |
