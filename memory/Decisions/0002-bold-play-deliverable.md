---
created: 2026-09-28
updated: 2026-09-28
tags: [decision]
status: accepted
---
# 0002 — Ship the bold-play challenge manager, report honestly

## Context
- User asked for a bot that passes 14-day challenges (+10 / −3 daily / −6)
  robustly incl. 2025–2026, and said not to report until it succeeds.
- Extensive research ([[Research/Findings]]) found no edge with the
  Sharpe ≥ 3 that a ≥50% pass rate requires. No-edge ceiling ≈ 37.5%.

## Decision
- Deliver `mt5/PropBoldPlay.mq5`: pass-optimal sizing + rule guards, with
  presets for XAUUSD and BTCUSD selected on train years only.
- Report the real out-of-sample pass rate (~33–37%) and state plainly that
  the success criterion was not met, instead of presenting an overfit result.
- Excluded rule exploits (gap gambling, cross-account hedging, latency arb).

## Consequences
- Future work only makes sense with new inputs: a real edge (other data,
  execution), different firm rules, or a longer time limit.
