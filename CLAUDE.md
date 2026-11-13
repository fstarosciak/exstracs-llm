# CLAUDE.md

Thesis project: ExSTraCS learning classifier system in Rust, seeded with LLM-generated rules.
Spec (Polish): `../specyfikacja-projektu-exstracs-llm.md`. Plan and roadmap: `docs/PLAN.md`.
Usage: `README.md`.

## Hard rules

- Never commit secrets: API keys, tokens, passwords, `.env` files, private keys, credentials,
  or personal data. Check staged files before any commit. If a secret is found in the working
  tree, warn the user and keep it out of git.
- Do not commit and do not push. The user commits and pushes. Only make a commit or a push when
  the user explicitly asks for that specific action, and only that once.
- Never delete anything from a remote: no `git push --force`, no deleting remote branches or
  tags, no history rewrites of pushed commits.
- Do not delete files or directories unless the user asks for it.

## Language and style

- Talk to the user in Polish, using caveman mode (terse, no filler, full technical content).
- Everything written to the repository is in English: code, identifiers, docs, README, plans,
  commit messages, test names, CLI help, JSON keys.
- No code comments (`//`, `///`), following the acs2-rust convention.

## Code conventions

- Rust edition 2021, stable toolchain; workspace root `Cargo.toml`, crate `exstracs/`.
- The crate must stay movable into the acs2-rust workspace: no dependency on `acs2-*` crates,
  same `clippy.toml`.
- Determinism from the seed: all randomness through the injected `ChaCha8Rng`; no `HashMap` or
  `HashSet` (use `BTreeMap`, `BTreeSet` or `Vec`); no `f64::powf`, `exp`, `ln` and similar
  (clippy.toml forbids them; use explicit multiplication).
- Implement from the ExSTraCS papers. scikit-ExSTraCS (GPL-3.0) is only a test oracle; never
  copy its code.
- Invalid seed rules are rejected and counted, never silently repaired.
- Run outputs go to `runs/` (ignored by git) or outside the repo, never into `src/` or `docs/`.

## Gates before reporting work as done

```sh
cargo build --release
cargo test
cargo test --release -- --ignored
cargo clippy --all-targets -- -D warnings
```

New code adds no clippy warnings. Report failing tests with their output; do not loosen test
thresholds to make them pass.

## Agents

When the user asks for delegation, use the cavecrew agents: `cavecrew-investigator` for
locating code, `cavecrew-builder` for 1–2 file edits, `cavecrew-reviewer` for reviews. Give
builders fixed signatures and an exact spec; they must not change public APIs owned by other
modules.
