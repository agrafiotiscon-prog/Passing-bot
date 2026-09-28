---
created: 2026-09-28
updated: 2026-09-28
tags: [research, data]
---
# Market data

Research data is fetched by `research/fetch_data.py` into `data/raw/`
(gitignored) and stitched by `research/data.py` into `data/bars/*.parquet`.

## Network
- The cloud container blocks market-data hosts (Dukascopy, Yahoo, Stooq,
  Binance, HistData, Kaggle, HuggingFace). GitHub (git + raw) and PyPI work,
  so all data comes from public GitHub repos.

## Coverage (server time, see below)
| Symbol | Span | Sources |
|---|---|---|
| XAUUSD M15 | 2012-05 → 2026-09 | MT5 exports (ejtraderLabs, APEX-JEV, dinesh), Dukascopy fill, getdata |
| EURUSD M15 | 2010-01 → 2026-09 | HistData M1, devffex MT5, getdata |
| GBPUSD, USDJPY M15 | 2012-11 → 2026-08/09 | ejtraderLabs, devffex MT5 (gap 2022-03→08) |
| NAS100/US30/SPX500 | 2020-09 → 2023-09 + 2026-03 → 09 | Dukascopy (SnowGuru), getdata |
| Other FX (AUD, CAD, CHF, crosses) | 2012-11 → 2022-03 | ejtraderLabs |

## Time zone — important
- Server time = `Europe/Helsinki` (UTC+2/+3, **EU** DST dates). The broker
  MT5 exports use this; verified bar-by-bar (corr ≥ 0.99 at lag 0 every month).
- HistData files are in **New York local time (with DST)**, despite docs
  saying fixed EST. Reading them as fixed EST put EURUSD 1h late every summer
  and created a fake cross-asset "edge" (Sharpe 2.6) via look-ahead. Always
  run the month-by-month lag check when adding a source.

## Costs used (per `research/harness.py`)
- Spread + slippage per fill, conservative: XAUUSD 0.30 + 0.10,
  EURUSD 1.0 + 0.3 pip, GBPUSD 1.4 + 0.4, USDJPY 1.2 + 0.4, NAS100 1.5 + 0.5.
