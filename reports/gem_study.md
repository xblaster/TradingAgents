# Gem rules event study 2023-01-01 → 2026-10-15

198 screens every 7 days over 10 assets. Excess = asset forward return minus the equal-weighted universe on the same window.

| rule | events | dates | excess 7d | t | p | hit 7d | excess 30d | t | p | hit 30d |
|---|---|---|---|---|---|---|---|---|---|---|
| All gems | 540 | 151 | -0.33% | -1.03 | 0.302 | 42% | -1.54% | -1.92 | 0.055 | 42% |
| G1 Emerging significant trend | 229 | 110 | +0.29% | +0.48 | 0.630 | 45% | -0.93% | -0.61 | 0.545 | 46% |
| G2 Volume-confirmed breakout | 58 | 27 | -1.50% | -1.23 | 0.224 | 40% | -3.94% | -1.17 | 0.247 | 33% |
| G3 Statistical capitulation | 2 | 1 | n/a | | | | n/a | | | |
| G4 Cointegration dislocation | 12 | 11 | -1.15% | -1.59 | 0.140 | 25% | +2.30% | +1.71 | 0.115 | 67% |
| G5 Volatility squeeze | 239 | 73 | -0.56% | -1.78 | 0.076 | 41% | -1.76% | -2.66 | 0.008 | 40% |

Reading guide: a rule is informative if its mean excess is positive with |t| well above 2 at several horizons.
Events overlap (weekly screens, 30-day horizons) so t-stats are optimistic; with 5 rules tested, expect one
spurious p < 0.05 about a quarter of the time. Treat this as evidence for or against routing gems to the LLM,
not as a trading signal on its own.
