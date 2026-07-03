#  Forecasting Stock Volatility

This project digs into how stock volatility behaves over time and how well it can be forecasted — using Apple (AAPL) as the case study, with about 10 years of daily price data.

The core of the project is data analysis: computing returns, measuring rolling volatility, understanding how it clusters over time, and benchmarking a couple of different forecasting approaches against each other, including the industry-standard GARCH model and an LSTM-based predictor.

## What this does

1. Cleans and processes 10 years of AAPL daily price data
2. Computes log returns and rolling (21-day) volatility
3. Builds features from returns, squared returns, volume change, and rolling volatility
4. Fits a GARCH(1,1) model — the classic statistical baseline used in quant finance for volatility forecasting
5. Also trains an LSTM model on the same features, mainly to see how it stacks up against the statistical baseline
6. Compares the two approaches on the same test period

## Results

| Model | RMSE | R² |
|---|---|---|
| GARCH(1,1) | 0.00367 | 0.618 |
| LSTM | 0.00136 | 0.948 |

The goal was to forecast volatility with the LSTM and see how it holds up against GARCH, the standard benchmark used in the industry. The LSTM came out ahead on both metrics, likely because it can pick up on longer-range patterns in how volatility clusters over time, something GARCH's fixed formula is more limited in capturing.

## Why this matters

Volatility forecasting is a genuinely practical problem in finance — it feeds into options pricing, risk management, and position sizing. GARCH has been the standard tool for this since the 1980s, so it makes a natural yardstick for testing whether other approaches add value.

## Tech stack

- **Data**: Historical AAPL prices (Kaggle)
- **Analysis & preprocessing**: pandas, NumPy, scikit-learn
- **Statistical modeling**: `arch` library (GARCH)
- **Forecasting comparison**: TensorFlow/Keras (LSTM, used as a secondary comparison point)
- **Evaluation**: RMSE, MAE, R²

## Notebook structure

1. Data loading & cleaning
2. Log return & rolling volatility calculation
3. Feature engineering + train/test split (temporal, not random — important for time series)
4. GARCH baseline fit
5. LSTM model training (comparison model)
6. Actual vs. predicted volatility comparison
7. Head-to-head evaluation

## A caveat, for transparency

The GARCH model here forecasts one-day-ahead conditional variance, while the LSTM predicts a 21-day rolling volatility measure. They're closely related but not identical targets, so the comparison should be read as directionally informative rather than a perfectly controlled experiment. Worth keeping in mind if you extend this.

## What I'd add next

- A regime classifier (high-vol vs. low-vol days) using Logistic Regression
- Testing on more tickers to check if results generalize beyond AAPL
- A cleaner apples-to-apples comparison between the two forecasting targets

---

*Built as a self-project to explore volatility behavior in markets, using both statistical and machine learning approaches.*
