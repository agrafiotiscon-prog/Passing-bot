---
created: 2026-09-28
updated: 2026-09-28
tags: [research, findings]
---
# Research findings log

Goal: bot that passes a 14-day prop challenge (+10% target, −3% daily,
−6% max loss) robustly across 2012–2026, incl. 2025–2026.
Simulator: `research/sim.py` (conservative intrabar ordering, costs).

## Theory
- Without an edge the best pass rate any sizing scheme can reach is
  ~6/16 = 37.5% (optional stopping); the 14-day limit lowers it further.
- Gaussian Monte Carlo (10 trading days, daily halt 2.5%): pass ≈ 35% at
  Sharpe 0, 45% at Sharpe 2, 50% at Sharpe 3, 60% at Sharpe 5, 75% at 8.
  ⇒ >50% needs a portfolio Sharpe ≥ 3 at high leverage.

## Tested, no robust edge after costs (2012–2026)
- Random entries (baseline): 1–17% pass depending on risk; mostly timeouts.
- Session range breakouts on gold (Asian, London, NY, 48 variants): ≤ 0.
- Donchian trend following on M15 (gold, EURUSD): negative/flat.
- Fade/follow extreme 1–4h moves (4 FX/gold symbols): negative.
- NAS100/US30/SPX500 5-min ORB (Zarattini-Aziz): +0.06R only on NAS100 RR10.
- Daily TSMOM single asset: yearly Sharpe flips, avg ~0.3–0.5.
- Hour-of-day seasonality: real but ~1 bp/h, below cost.
- Market intraday momentum (open→16:30 predicts rest of day): gold ~2 bp/day,
  t≈2.2 — marginal.
- Walk-forward LightGBM (hourly, 4 symbols, ±cross-asset): flat/negative
  after fixing the HistData time-zone leak.
