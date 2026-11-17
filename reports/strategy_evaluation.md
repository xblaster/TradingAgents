# Revolut X trend strategy: pre-registered evaluation

## Verdict: **NOT VIABLE**

Protocol, parameters and criteria: [STRATEGY.md](../STRATEGY.md), committed before this evaluation was run.

| | Criterion | Result | Detail |
|---|---|---|---|
| C1 | CAGR > 0 after costs (core and extended) | PASS | core +16.7%, extended +11.3% |
| C2 | Sharpe >= equal-weight buy & hold (core and extended) | **FAIL** | core 0.64 vs 0.72, extended 0.49 vs 0.54 |
| C3 | Max drawdown <= 0.6 x equal-weight buy & hold (core and extended) | **FAIL** | core -36.3% vs limit -35.6%, extended -48.2% vs limit -45.9% |
| C4 | >= 75% of 12 parameter variants pass C2 and C3 (core) | **FAIL** | 5/12 pass (42%) |
| C5 | CAGR > 0 with doubled costs, 30 bps (core and extended) | PASS | core +15.5%, extended +10.3% |
| C6 | In-sample 2018-2022 sanity: Sharpe >= and drawdown < equal-weight buy & hold (core) | PASS | Sharpe 0.78 vs 0.56, max DD -67.7% vs -86.3% |

## Out-of-sample 2023 → today, core (BTC, ETH)

| | CAGR | total | vol | Sharpe | max DD | Calmar | exposure | turnover/yr | fees |
|---|---|---|---|---|---|---|---|---|---|
| Trend portfolio (15 bps) | +16.7% | +81.9% | +31.8% | 0.64 | -36.3% | 0.46 | 59% | 7.0x | +4.1% |
| Equal-weight buy & hold | +26.8% | +151.4% | +51.8% | 0.72 | -59.3% | 0.45 | 100% | 0.7x | +0.4% |
| BTC buy & hold | +40.9% | +278.7% | +46.1% | 0.97 | -51.8% | 0.79 | 100% | 0.3x | +0.1% |
| Trend portfolio (30 bps) | +15.5% | +74.6% | +31.8% | 0.61 | -37.1% | 0.42 | 59% | 7.0x | +8.2% |

## Out-of-sample 2023 → today, extended (10 pairs)

| | CAGR | total | vol | Sharpe | max DD | Calmar | exposure | turnover/yr | fees |
|---|---|---|---|---|---|---|---|---|---|
| Trend portfolio (15 bps) | +11.3% | +51.5% | +33.6% | 0.49 | -48.2% | 0.23 | 47% | 6.1x | +3.5% |
| Equal-weight buy & hold | +15.3% | +73.6% | +62.8% | 0.54 | -76.4% | 0.20 | 98% | 0.9x | +0.5% |
| BTC buy & hold | +40.9% | +278.7% | +46.1% | 0.97 | -51.8% | 0.79 | 100% | 0.3x | +0.1% |
| Trend portfolio (30 bps) | +10.3% | +46.3% | +33.6% | 0.46 | -49.0% | 0.21 | 47% | 6.1x | +7.1% |

## In-sample 2018-2022, core

| | CAGR | total | vol | Sharpe | max DD | Calmar | exposure | turnover/yr | fees |
|---|---|---|---|---|---|---|---|---|---|
| Trend portfolio (15 bps) | +29.8% | +268.1% | +49.2% | 0.78 | -67.7% | 0.44 | 53% | 6.8x | +5.1% |
| Equal-weight buy & hold | +13.4% | +87.4% | +79.4% | 0.56 | -86.3% | 0.16 | 100% | 1.1x | +0.8% |
| BTC buy & hold | +6.3% | +35.8% | +72.9% | 0.45 | -80.4% | 0.08 | 100% | 0.2x | +0.1% |

## In-sample 2018-2022, extended

| | CAGR | total | vol | Sharpe | max DD | Calmar | exposure | turnover/yr | fees |
|---|---|---|---|---|---|---|---|---|---|
| Trend portfolio (15 bps) | +57.8% | +878.5% | +55.7% | 1.10 | -64.1% | 0.90 | 49% | 6.8x | +5.1% |
| Equal-weight buy & hold | +43.7% | +512.1% | +89.8% | 0.86 | -83.2% | 0.52 | 99% | 1.8x | +1.3% |
| BTC buy & hold | +6.3% | +35.8% | +72.9% | 0.45 | -80.4% | 0.08 | 100% | 0.2x | +0.1% |

## Calendar years, out-of-sample, core

| | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| Trend portfolio (15 bps) | +46.3% | +51.1% | -0.7% | -17.2% |
| Equal-weight buy & hold | +114.3% | +94.3% | -16.5% | -27.7% |
| BTC buy & hold | +146.2% | +135.7% | -17.4% | -21.0% |

## Statistical significance, out-of-sample, core (informational: does not change the verdict)

| | value |
|---|---|
| Strategy PSR (P[true Sharpe > 0]) | 0.90 |
| Strategy Deflated Sharpe (13 configurations tested) | 0.82 |
| Strategy Sharpe, 90% stationary-bootstrap CI | 0.64 [-0.23, 1.47] |
| Strategy CAGR, 90% CI | +16.7% [-11.0%, +54.8%] |
| Track record needed for 95% confidence in Sharpe > 0 | 2366 days (6.5 years) |
| EW buy & hold PSR | 0.92 |
| EW buy & hold Sharpe, 90% CI | 0.72 [-0.13, 1.64] |
| Sharpe(strategy) - Sharpe(EW), 90% CI | -0.07 [-0.55, +0.33], P(strategy better) = 36% |

## Parameter robustness (C4), out-of-sample, core

| lookbacks | rebalance | CAGR | Sharpe | max DD | C2 & C3 |
|---|---|---|---|---|---|
| (10, 30, 60, 125) | 3d | +24.6% | 0.88 | -31.0% | pass |
| (10, 30, 60, 125) | 7d | +24.3% | 0.87 | -29.1% | pass |
| (10, 30, 60, 125) | 14d | +22.8% | 0.82 | -29.6% | pass |
| (15, 45, 90, 188) | 3d | +20.3% | 0.74 | -31.4% | pass |
| (15, 45, 90, 188) | 7d | +15.3% | 0.60 | -35.0% | fail |
| (15, 45, 90, 188) | 14d | +14.5% | 0.57 | -35.1% | fail |
| (25, 75, 150, 312) | 3d | +16.8% | 0.65 | -30.9% | fail |
| (25, 75, 150, 312) | 7d | +15.6% | 0.62 | -29.6% | fail |
| (25, 75, 150, 312) | 14d | +14.9% | 0.60 | -30.4% | fail |
| (30, 90, 180, 375) | 3d | +19.2% | 0.71 | -30.6% | fail |
| (30, 90, 180, 375) | 7d | +20.2% | 0.73 | -30.4% | pass |
| (30, 90, 180, 375) | 14d | +19.0% | 0.68 | -35.1% | fail |

## Target weights at the last close (extended universe)

- LINK-EUR: 4.3%
- BTC-EUR: 4.0%
- AVAX-EUR: 1.6%
- EUR: 90.2%

## Notes

- Data: Yahoo daily bars, 2016-04-11 to 2026-11-17; the out-of-sample window ends 2026-11-17.
- Universe chosen among pairs listed on Revolut X today: survivorship bias favours the extended universe.
