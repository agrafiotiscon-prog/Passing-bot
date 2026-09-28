---
created: 2026-09-28
updated: 2026-09-28
tags: [decision]
status: accepted
---
# 0001 — Use an Obsidian vault for memory

## Context
- The user asked for Obsidian to be used for memory.
- Claude Code sessions run in ephemeral cloud containers. Only what is
  committed to the repo survives between sessions.
- There is no Obsidian connector in the session, so a local vault on the
  user's machine can't be reached directly.

## Decision
- Keep memory as an Obsidian vault in the repo at `memory/`.
- `CLAUDE.md` imports `memory/Home.md`, so the index loads automatically
  every session.
- Obsidian settings shared through git: `app.json`, `templates.json`,
  `daily-notes.json`. Per-device workspace state is gitignored.

## Consequences
- The user opens `memory/` in Obsidian ("Open folder as vault") after
  pulling. The Obsidian Git community plugin can keep it synced.
- Memory changes go through git like code, so they have history and review.
