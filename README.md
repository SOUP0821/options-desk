# Options desk

Black-Scholes pricer (price, Greeks, implied vol) plus one SPY backtest. The pricing checks are published textbook values, finite differences, a torch autograd copy of the formula, and Monte Carlo. The backtest rules were written down before the 2019-2026 window was scored.

## Run

From this directory, with the packages in `requirements.txt`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
python -m options_desk
```

`pytest` does not use the network. The report downloads the SPX option chain, the 13-week bill, and adjusted SPY history, then writes `results/summary.json`, `results/chain.csv`, and `results/test_returns.csv`. A cached SPY file is kept at `data/spy.csv` and used only if the download fails.

The numbers below are from the run on 23 October 2026.

## Pricer

`price` and `greeks` are the Black-Scholes-Merton formulas for a European call or put. Volatility is the annualized standard deviation of log returns. Rates and dividend yields are continuous. Vega is the derivative with respect to volatility, not the change for one volatility point. Theta is the derivative with respect to calendar time, in price per year.

Published prices, including Hull example 15.6 (spot 42, strike 40, 10% rate, 20% volatility, half a year):

| Check | Result |
| --- | --- |
| Hull call 4.759422 | error 3.9e-7 |
| Hull put 0.8085994 | error -2.7e-8 |
| Largest error on the five published cases | 4.6e-6, inside the last printed digit |
| Hull 20-week call, textbook 2.40 / delta 0.522 | price 2.4005, delta 0.5216, gamma 0.0655, vega 0.1211 per point, theta -4.31 per year, rho 0.0891 per point |
| Largest gap versus a central finite difference | vega 2.0e-6; delta 7.2e-11 |
| Implied-vol round trip on a 240-point grid | max error 6.7e-12 |
| Antithetic Monte Carlo, 200,000 paths, Hull call | 4.7504 versus 4.7594, standard error 0.0055 |
| PyTorch autograd versus the analytical Greeks | about 1e-14; the price matches the published cases to the printed digit |

Put-call parity is an identity of the formula, and call and put share gamma and vega. Expired and zero-volatility limits are tested separately.

## Live chain

SPX, which is European, on 23 October 2026. Spot 7,623.68. Expiry 23 November 2026, 31 calendar days, so time is 31/365. The rate is the 13-week bill discount quote, 4.032%, converted to a continuous yield of 4.11% and applied flat. The dividend yield, 0.50%, is the median put-call parity yield from 11 strikes within 5% of spot.

Bids and asks were zero after the close, so the run used last trades (the rule fixed beforehand: use mids only when at least eight calls have a usable market). Fifty-eight of 59 quotes inside 80–120% of spot had a finite volatility. One 8000 put, last 337, sat below the European lower bound, so it has no volatility. The median implied volatility was 15.10%. The median absolute gap between the call and the put at the same strike was 0.22 volatility points.

The 7625 call last-traded at 144.40. Inverting that premium gives 15.05% volatility, and putting that volatility back through the formula reproduces 144.40 (residual about 1e-13). A 300,000-path Monte Carlo at that volatility priced it at 144.25 with a standard error of 0.27.

Yahoo's `impliedVolatility` field on this snapshot has median 1.56%. That column is not a second calculation of these last trades. The useful external check on this chain is call/put agreement, the inversion residual, and the Monte Carlo standard error.

## Backtest

Adjusted SPY, dividends reinvested. A signal at the close earns the next session's return. Cash earns zero. Sharpe uses a zero risk-free rate and 252-day annualization. Costs are charged on turnover, including the trade that opens a flat book. One basis point is the cost used to choose the threshold. Five basis points is a stress of those same positions.

The split is fixed: fit through 2014, choose the threshold on 2015–2018, score 2019 once.

- Buy and hold.
- 12-1 momentum: long when the return from 252 sessions ago to 21 sessions ago is positive, otherwise flat. No estimated parameters.
- L2 logistic regression on trailing 1-, 5-, 21-, and 63-day returns, 21-day realized volatility, and distance from the 21-day average. The scaler is fit on the training rows only. The label is the sign of the next day's return. The long/flat threshold is chosen from {0.50, 0.52, 0.55} by validation Sharpe, and a tie takes the higher threshold.

The fit used 4,024 training rows, 1,005 validation rows, and 1,963 test signals from 2 January 2019 through 22 October 2026. The chosen threshold was 0.50. Training accuracy was 54.15%, against a 54.08% base rate of up days. Validation Sharpes were 0.608, 0.514, and 0.154. The standardized coefficients are small; the largest is -0.10 on the 21-day return.

Test window, wealth starting at 1:

| Strategy | Cost | CAGR | Vol | Sharpe | Max drawdown | Time in market | Trades | Wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Buy and hold | 1 bp | 17.03% | 19.18% | 0.92 | -33.72% | 100% | 1 | 3.40 |
| 12-1 momentum | 1 bp | 13.52% | 17.33% | 0.82 | -33.72% | 84.7% | 39 | 2.69 |
| Logistic | 1 bp | 17.12% | 18.39% | 0.95 | -31.45% | 95.0% | 77 | 3.42 |
| Buy and hold | 5 bp | 17.02% | 19.18% | 0.92 | -33.72% | 100% | 1 | 3.40 |
| 12-1 momentum | 5 bp | 13.30% | 17.33% | 0.81 | -33.72% | 84.7% | 39 | 2.64 |
| Logistic | 5 bp | 16.65% | 18.39% | 0.93 | -31.51% | 95.0% | 77 | 3.32 |

The logistic book is long on 95% of test days. At 1 bp its CAGR is 17.12% and the hold is 17.03%. At 5 bp the logistic CAGR is 16.65%. Twelve-one momentum finishes at 13.52% and 13.30%. That is the holdout. The threshold and the features were not changed after these numbers were computed.

## Tests

`tests/` locks the published prices, put-call parity, finite-difference Greeks, implied-vol round trips, Monte Carlo sampling error, the PyTorch derivatives, the bill-yield conversion, the expiry rule, recovery of a known volatility from a synthetic chain, and the backtest split. The threshold test builds a validation window that wants 0.55 and a test window that wants 0.50, and checks that 0.55 is chosen. Feature rows are checked to be unchanged when future prices change, while the next-day label does change.
