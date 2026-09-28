---
created: 2026-09-28
updated: 2026-09-28
tags: [project]
---
# Project overview

## Purpose
- Build an automated MT5 bot that passes prop-firm evaluation challenges:
  +10% profit target, −3% max daily loss, −6% max overall loss, ~14-day time
  limit. Demo accounts: no commissions, small slippage.
- Must be validated on recent data (incl. 2025–2026), not just old periods.

## Current state
- 2026-09-28: research done ([[Research/Findings]], [[Research/Data]]).
  No robust edge found; shipped `mt5/PropBoldPlay.mq5` (bold-play sizing +
  rule guards), ~33–37% pass rate per 14-day attempt out of sample
  ([[Decisions/0002-bold-play-deliverable]]). EA not yet compiled/tested in MT5.

## Layout
- `research/` — Python: data fetch/stitch, numba challenge simulator,
  strategies, ML walk-forward.
- `mt5/` — Expert Advisor `PropBoldPlay.mq5` + presets for XAUUSD, BTCUSD.
- `data/` — downloaded bars (gitignored; rebuild with `research/fetch_data.py`
  then `research/data.py`).

## Related
- [[Decisions/0001-obsidian-memory]]
