# Tail-Risk Forecasting

## Status
Work in progress. Done: data pipeline and checks, volatility proxies, historical-simulation VaR/ES with Kupiec backtest.
Next: Christoffersen independence test, GARCH and HAR-based VaR, conformal calibration and e-backtesting.

## Goal

Forecast 1-day Value-at-Risk (VaR) and Expected Shortfall (ES) for an equity index, calibrate them with conformal methods, and evaluate them with e-backtests.

## Data

Data consists of daily open, high, low and close prices for the S&P 500 index, sourced from Yahoo Finance via `yfinance`, from January 2015 to October 2026. Daily log returns are computed from the close price, and the loss is defined as the negative of the daily return, so that a positive value is a loss, the convention for VaR and ES.

Before any modelling, each price series is passed through `check_prices`, which counts missing values, non-positive prices, and days where the high is below the low, the close is outside the daily range, or the high equals the low (a sign of stale quotes), as well as duplicated or unordered dates. The S&P 500 sample passes all checks. Impossible rows (e.g. a high below the low) would be removed, but extreme returns are kept, because they are the most informative for risk management and test the limits of traditional VaR and ES methods.

A single squared daily return is an unbiased but very noisy measure of that day's variance. As a less noisy proxy, I also compute the range-based estimator of Parkinson (1980), as presented in Christoffersen (2012, Section 6.1). The estimator assumes normally distributed log returns with zero mean within the day, and only uses prices observed during trading hours.

$$\hat\sigma^2_t = \frac{1}{4\ln 2}\left(\ln\frac{H_t}{L_t}\right)^2$$

where $H_t$ and $L_t$ are the daily high and low.

The figure compares 21-day rolling annualised volatility from both measures. They follow the same regimes, but Parkinson is consistently lower, because the high–low range only covers trading hours and misses the overnight gap between the previous close and the open. This gap is largest in crises: in March 2020, close-to-close volatility reached about 95%, against roughly 53% for Parkinson. Extending the proxy with estimators that include the overnight return (Garman–Klass, Yang–Zhang) is a next step.

![Close-to-close vs Parkinson volatility](results/losses_and_vol.png)
## Method

Planned.

## Results

To come.

## References

- Christoffersen, P. F. (2012). *Elements of Financial Risk Management* (2nd ed.). Academic Press.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61–65.