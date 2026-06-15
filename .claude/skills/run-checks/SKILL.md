---
name: run-checks
description: Run scripts/checks.sh (lint, coverage, integration tests, docs, translations, copyrights) repeatedly, diagnosing and fixing failures between runs, until it passes cleanly. Use when the user wants the full pre-push check suite to pass.
---

# Run checks until green

Run `scripts/checks.sh` and keep iterating — fix what fails, re-run — until it exits 0.

## What the script does

`scripts/checks.sh` runs the full pre-push suite. It uses `set -euo pipefail`,
sources `.venv/bin/activate`, and writes everything to `scripts/checks.log`. In order:

1. `scripts/linting.sh` — linting and formatting
2. `scripts/coverage.sh` — compile and coverage
3. `pytest tests/integration` — integration tests
4. `scripts/docs.sh` — docs
5. `scripts/missing_translations.sh` — missing translations
6. `scripts/copyrights.sh` — copyright headers

It prints `All checks passed successfully!` only when every stage passes.

## Loop

1. Run `bash scripts/checks.sh` from the repo root (`/home/luke/Documents/qilisdk`).
2. If it exits 0 — done. Report success.
3. If it fails:
   - Read the tail of `scripts/checks.log` (and the stage-specific log, e.g. `scripts/cov.log`)
     to find the first failing stage. Because of `pipefail`, the first failure aborts the run.
   - Diagnose the root cause and apply the smallest correct fix.
   - Re-run `bash scripts/checks.sh`.
4. Repeat until it passes.

## Guidance

- Fix the actual cause; never weaken a check, skip a test, or lower coverage thresholds just to
  make it pass. If a check looks genuinely wrong or a fix is ambiguous, stop and ask the user.
- Always run the full check suite after each set of fixes, even if the failure was in an early stage. This ensures that fixes don't cause new failures later in the pipeline and prevents having to ask the user to run the subscripts etc. manually.
- Don't commit or push.
