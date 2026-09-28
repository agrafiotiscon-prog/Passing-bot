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
- Anomaly scan, ~19k rules, discovery 2012–19 / validation 2020–22 / OOS
  2023–26 (`research/scan.py`): survivors were rollover-spread artifacts or
  reversed OOS. Under raw costs: EURUSD down in EU hours (Breedon-Ranaldo),
  decayed to ~0.7 bp/day by 2023–26.
- Session-drift portfolio (EUR/GBP short, JPY long in EU hours, gold long
  Asia): Sharpe 1.14 (2012–19) → 0.13 (2020–26).
- Daily MR: RSI(2)<10 long on NAS100/SPX500 ~+31 bp/trade, t≈2.4, 24/yr,
  bull-market dependent. Nothing on FX/gold.
- BTC (Bitstamp 2016–2026): 20d TSMOM Sharpe 1.27, negative 2025. Intraday
  effects nil.

## Challenge-structure results (the deliverable)
- "Bold play" (1 entry/day, stop k×ATR_d, TP sized so a win = +10%, risk
  min(2.9%, room)) is the pass-optimal sizing: coin-flip direction passes
  35.3% (gold) / 37.5% (BTC), ≈ the 37.5% theoretical ceiling.
- Direction filters chosen on train years did not beat coin flip OOS:
  gold 33.5% vs 36.2% (2020–26), BTC 37.1% vs 35.9% (2022–26).
- Non-passes end parked at 94.2% equity after two full losses.
- Longer windows only help with real edge: BTC trend, 365-day window,
  2% risk → 45% pass since 2022.
- Conclusion: reliable passing (≥50%) needs Sharpe ≥ 3; none found. See
  [[Decisions/0002-bold-play-deliverable]].
