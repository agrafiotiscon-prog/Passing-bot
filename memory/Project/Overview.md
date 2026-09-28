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
- 2026-09-28: data pipeline, challenge simulator and first strategy scans done
  (see [[Research/Findings]], [[Research/Data]]). No robust edge found yet.

## Layout
- `research/` — Python: data fetch/stitch, numba challenge simulator,
  strategies, ML walk-forward.
- `data/` — downloaded bars (gitignored; rebuild with `research/fetch_data.py`
  then `research/data.py`).

## Related
- [[Decisions/0001-obsidian-memory]]
