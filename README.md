# Passing-bot

An MT5 Expert Advisor and the research behind it for prop-firm evaluation
challenges with these rules: **+10% target, −3% max daily loss, −6% max overall
loss, 14-day limit**, on a demo account with no commissions.

## Bottom line

**No strategy in this research passes such a challenge reliably.** The best
legitimate configuration passes about **1 in 3 attempts** (gold ≈ 34–36%,
BTC ≈ 36–37% out of sample). Every year 2012–2026 lands between 28% and 59%
under raw costs, 2025–2026 included.

That pass rate comes from sizing that is optimal for the target and limit
structure, not from a market edge. The EA is not a profitable trading
strategy for a funded account.

Why higher is not reachable here:

- **The math.** With no edge, no sizing scheme can reach +10% before −6% more
  than about 6/16 ≈ **37.5%** of the time (optional stopping). A Monte Carlo
  with the daily limit and 10 trading days gives ≈35% at Sharpe 0, 45% at
  Sharpe 2, 50% at Sharpe 3 and 60% at Sharpe 5. So passing reliably needs a
  strategy with an annualised Sharpe of about 3 or more.
- **The evidence.** Across gold, EURUSD, GBPUSD, USDJPY, US indices and BTC for
  2012–2026 (details in `memory/Research/Findings.md`), nothing robust exceeded
  a Sharpe of about 1 after costs out of sample. Tested:
  - roughly 20,000 time-of-day and conditional rules
  - walk-forward LightGBM
  - session and opening-range breakouts
  - trend following and time-series momentum
  - daily mean reversion (IBS, RSI(2))
  - intraday momentum
  - session drifts
  - gap and large-range-day effects

  Every effect that looked good in-sample decayed or reversed out of sample.
  An apparent Sharpe 2.6 ML edge on EURUSD turned out to be a time-zone
  look-ahead bug in a data source, and was fixed.
- **Round 2, per-instrument strategies across 26 instruments** (FX majors
  and crosses, gold, silver, Brent, BTC and 8 altcoins, US indices; see
  `research/library.py`, `pairs.py`, `meta.py`):
  - a 1,770-rule library chosen per instrument on training years
  - pairs mean reversion
  - walk-forward "strategy momentum" selection
  - liquidity-sweep reversals
  - NFP drift
  - cross-sectional momentum and reversal

  The best honest out-of-sample portfolio reached a Sharpe of about 0.3–0.5.
  Several apparent Sharpe 1.3–4.5 edges turned out to be data artifacts and
  were removed: bid prices around the daily rollover and gold's re-open, and
  the high/low of synthetic crosses.

## The EA: `mt5/PropBoldPlay.mq5`

"Bold play" is the pass-probability-optimal way to take few, large bets.

- One market entry per day at a fixed server time. The direction comes from a
  trend filter: the sign of the last N daily closes.
- The stop is k × daily ATR(14).
- The take-profit is placed so that one win lifts equity straight to +10%.
- Risk per trade is min(2.9%, room left to the daily halt, room left to the
  overall floor). Two full losses leave the account at about −5.8%, where it
  stops trading.
- Guards run every tick and every second:
  - target lock: close everything at +10% and stop trading
  - daily equity halt at −2.9% from the day's reference (max of balance and
    equity at the reset time)
  - overall floor
  - 14-day timer
- Flat before the daily rollover.

State is kept in terminal global variables, so a restart doesn't lose the
challenge's start balance or date. The code has not been compiled here, so
compile it in MetaEditor and run it in the Strategy Tester before going live.

Presets are in `mt5/presets/`:

| Preset | Entry | Direction | Stop | Notes |
|---|---|---|---|---|
| `XAUUSD.set` | 10:00 server | 10-day trend | 0.2 × ATR | selected on 2012–2019 |
| `BTCUSD.set` | 01:00 server, 7 days/week | 3-day trend | 0.4 × ATR | selected on 2016–2021; needs ≥1:3.3 leverage |

## Backtest results

Pass rate per 14-day window, one window starting on each (trading) day. Costs
are raw spread plus slippage (gold $0.15 + $0.05 per fill, BTC 3 bp + 1 bp).
Runs are on M15 bars, and M5 bars give the same numbers within 0–2 points.
Out-of-sample years are marked ▶.

| Year | Gold (preset) | Gold, coin-flip direction | BTC (preset) | BTC, coin-flip direction |
|---|---|---|---|---|
| 2012 | 39.7% | 35.4% | – | – |
| 2013 | 38.8% | 37.3% | – | – |
| 2014 | 39.9% | 31.1% | – | – |
| 2015 | 43.4% | 37.1% | – | – |
| 2016 | 48.4% | 30.4% | 41.5% | 42.9% |
| 2017 | 42.4% | 34.6% | 59.2% | 38.4% |
| 2018 | 32.9% | 36.8% | 42.2% | 43.1% |
| 2019 | 28.7% | 34.7% | 46.8% | 34.6% |
| 2020 | ▶ 39.8% | 37.1% | 43.2% | 41.2% |
| 2021 | ▶ 38.4% | 36.7% | 32.1% | 33.2% |
| 2022 | ▶ 28.7% | 36.2% | ▶ 34.5% | 32.9% |
| 2023 | ▶ 28.4% | 33.3% | ▶ 37.5% | 38.4% |
| 2024 | ▶ 29.3% | 30.3% | ▶ 43.7% | 36.5% |
| 2025 | ▶ 41.9% | 39.2% | ▶ 37.5% | 34.4% |
| 2026 (to Sep) | ▶ 28.2% | 40.5% | ▶ 32.3% | 37.2% |
| **Out-of-sample avg** | **33.5%** | 36.2% | **37.1%** | 35.9% |

- **Direction filter.** Out of sample, the trend filter did no better than a
  coin flip (gold 33.5% vs 36.2%, BTC 37.1% vs 35.9%). That is the clearest
  sign that the pass rate is set by the sizing, not by prediction.
- **Conservative costs.** With a spread of $0.30 + $0.10 per fill on gold, the
  full-period pass rate drops from 36.7% to 33.9%, and BTC from 41.2% to 38.0%.
- **How challenges end.** Passes come fast: the median is 1.7 days for gold and
  3.5 days for BTC. About 98% of gold non-passes and 92% of BTC non-passes end
  parked at about −5.8% after two full losses, unable to trade on.
- **Economics.** Expected value per attempt ≈ 0.35 × (value of a funded
  account) − fee. The funded account itself has no edge behind it.

## Why no rule exploits

Some things raise pass rates by breaking typical prop-firm terms, and were
deliberately left out:

- gambling on weekend or news gaps to overshoot the loss limit
- hedging across multiple accounts
- latency arbitrage against the demo feed

They also don't carry over to funded accounts.

## What would change the answer

- **A genuine edge with Sharpe ≥ 3.** None was found in public OHLC data at
  retail execution.
- **A longer or unlimited time limit.** This only helps with a real
  positive-expectancy strategy. Even with a 365-day window, BTC 20-day trend
  following passes about 45% of windows since 2022.
- **Wider rules.** A larger overall-loss allowance relative to the target
  raises the no-edge ceiling, which is (max loss) / (max loss + target).

## Reproduce

```bash
pip install pandas numpy numba pyarrow scipy lightgbm scikit-learn
python research/fetch_data.py         # public GitHub data -> data/raw
python research/data.py               # stitch to MT5 server time -> data/bars
cd research && python final_eval.py   # train-period selection + yearly report
```

- Simulator: `research/sim.py`. It applies the conservative intrabar order
  (stop before target), a simultaneous-adverse-extreme check for the equity
  limits, and spread plus slippage on every fill.
- Strategy scans: `strategies.py`, `scan.py`, `ml.py`, `ml_eval.py`.
- Research log and data notes: the Obsidian vault in `memory/`.
