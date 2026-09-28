# Passing-bot

## Memory: Obsidian vault

Long-term memory for this project lives in the Obsidian vault at `memory/`.
Sessions run in ephemeral containers, so anything not written to the vault and
committed is lost when the session ends.

The vault index is loaded below on every session:

@memory/Home.md

### At the start of a session

- Read the notes linked from Home that bear on the task — always
  `memory/User/Preferences.md` and `memory/Project/Overview.md`.

### Before finishing a session

- Log the session in `memory/Sessions/YYYY-MM-DD.md` (create it from
  `memory/Templates/Session.md`, or append a new section if it already exists).
- Record each significant decision as `memory/Decisions/NNNN-short-title.md`
  from `memory/Templates/Decision.md`, numbering sequentially.
- Update `Project/Overview.md` and `User/Preferences.md` when facts change.
  Edit existing notes rather than duplicating them.
- Add every new note to `memory/Home.md`.
- Commit vault changes along with the code they relate to.

### Conventions

- Link notes with path-style wikilinks, e.g. `[[Project/Overview]]`.
- Frontmatter on every note: `created`, `updated` (YYYY-MM-DD), `tags`.
- Keep notes short and factual; prefer bullets.
- Never store secrets, tokens, or credentials in the vault.
