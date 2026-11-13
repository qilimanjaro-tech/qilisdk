# qilisdk

QiliSDK is Qilimanjaro's open-source Python framework, with a C++ simulator (QiliSim), for writing digital, analog and hybrid quantum algorithms and running them on CPU, GPU or QPU backends. It is public, Apache-2.0 and published on PyPI.

## Navigate by task
- Problems, variables, expressions, `QTensor` and results: `src/qilisdk/core/`. Hamiltonians and schedules: `src/qilisdk/analog/`.
- Gates, circuits, ansatz and transpiler: `src/qilisdk/digital/`, passes in `src/qilisdk/digital/circuit_transpiler_passes/`.
- What a backend executes: functionals in `src/qilisdk/functionals/`, readout in `src/qilisdk/readout/`, cost functions in `src/qilisdk/cost_functions/`, optimizers in `src/qilisdk/optimizers/`. Noise: `src/qilisdk/noise/`.
- Backends: `src/qilisdk/backends/` (QiliSim, plus the optional CUDA-Q and QuTiP). QiliSim's C++ core: `src/qilisdk_cpp/`, built by `CMakeLists.txt` through pybind11.
- Interop, solvers and plotting: `src/qilisdk/utils/`. SpeQtrum client: `src/qilisdk/speqtrum/`. ML datasets: `src/qilisdk/ml/`. Optional-dependency loading: `src/qilisdk/_optionals.py`.
- User docs: Sphinx `.rst` under `docs/` (a page per module in `docs/modules/`, tutorials in `docs/tutorials/`), translated in `docs/locale/`.
- Tests: `tests/unit_python/`, mirroring `src/qilisdk/`; `tests/integration/` (cross-backend runs); C++ GoogleTest suite in `tests/unit_cpp/`.
- Contribution workflow and the AI-tools policy: `CONTRIBUTING.md`. Local scripts: the "Development Guide" in `README.md` and `scripts/`.
- Before writing or reviewing code or tests: read `CODING_STANDARDS.md` and `REVIEW.md`.
- Domain vocabulary: `GLOSSARY.md`; name code, tests and PRs with its terms. Decisions: `docs/adr/`. Tracker and triage: `docs/agents/`.
- Project knowledge base: TODO(human): link the domain wiki.

## Commands
- Setup: `uv sync --frozen --python 3.13 --group dev --group backends --extra speqtrum --extra openfermion --extra qir --extra openqasm --extra scip`. It compiles QiliSim, so it needs a C++ compiler. CI adds `--extra all-cu13`, whose CUDA wheels exist for Linux and Windows only. It pins Python 3.13, as `scripts/install.sh` and CI's Types job do, because CUDA-Q has no 3.14 wheels (`.python-version` says 3.14) and ty needs it to resolve `cudaq`.
- Focused test: `uv run --no-sync pytest tests/unit_python/core/test_qtensor.py`, or `uv run --no-sync pytest tests/unit_python -k permute`.
- **Verify (run before calling work done):** `uv sync --frozen --python 3.13 --group dev --group backends --extra speqtrum --extra openfermion --extra qir --extra openqasm --extra scip --reinstall-package qilisdk && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync ty check --python-version 3.11 && uv run --no-sync ty check --python-version 3.12 && uv run --no-sync ty check --python-version 3.13 && uv run --no-sync ty check --python-version 3.14 && uv run --no-sync pytest -n auto tests/unit_python/`
  (the local part of the Code Quality, Tests and Platform Builds workflows. Not reproduced: clang-format on changed C++ files, the C++ test suite, the lowest-dependency jobs, the other Python versions and operating systems, the CUDA extra and Codecov's patch target. Report those as not run.)

## Boundaries
- Validate with synthetic inputs and offline tests; hardware, lab databases and production runcards are in scope only when the task names them.
- Report every gate you could not run and why.
- Keep credentials, runcards with real addresses and measurement data out of the repository and out of agent instructions.

## Critical invariants
- The repository is public: keep internal links, hostnames, customer names and credentials out of code, docs, commits and PRs, because anyone can read them.
- A human opens and owns every PR and answers its review, because `CONTRIBUTING.md` closes PRs that agents or bots open on their own.

<important if="you edit `src/qilisdk_cpp/`, `CMakeLists.txt` or `tests/unit_cpp/`">
- The editable install does not rebuild the extension: rerun the Setup command with `--reinstall-package qilisdk` (the verify command does) before testing from Python.
- C++ tests: add `-Ccmake.define.tests=ON` to that command, then run `./tests/unit_cpp/test_cpp`. The test target links `gcov` unconditionally (`CMakeLists.txt`), so it builds with GCC on Linux, as in CI, and fails to link on macOS.
- Formatting follows `.clang-format`: `clang-format --dry-run --Werror <file>`, which CI's Cpp Linter runs on the changed files. clang-tidy runs in a build with `-Ccmake.define.tidy=ON`.
- A new `.cpp` or `.h` file carries the Apache header naming Qilimanjaro Quantum Tech (`bash scripts/copyrights.sh`), because ruff's copyright rule covers Python only.
</important>

<important if="you add or change a dependency, an extra or a dependency group">
- The core install stays small: a backend or interop library goes behind an extra or the `backends` group and is imported lazily through `src/qilisdk/_optionals.py`, because `pip install qilisdk` must work without it (Platform Builds installs no extras).
- A floor in the `backends` group is mirrored in `OPTIONAL_FEATURES` in `src/qilisdk/backends/__init__.py`, which enforces it at runtime. A floor raised for a specific reason carries a comment with that reason, as in `pyproject.toml`.
- Update the lockfile with `uv lock` in the same PR: CI installs exactly what `uv.lock` holds (`--frozen`), so a change missing from it goes untested.
</important>

<important if="you add or change a public symbol, or edit `docs/`">
- A public symbol goes into its package's `__all__` and its `__init__.pyi` stub, which `tests/unit_python/test_stubs.py` compares, and into its page under `docs/modules/`.
- Python code blocks in `docs/`, changelog fragments and docstrings run under `bash scripts/docs.sh`; CI does not run them. Changed docs text leaves the Spanish and Catalan catalogs in `docs/locale/` stale: `bash scripts/missing_translations.sh` lists them.
</important>

<important if="you touch numeric code, units, golden values or tolerances">
- Units: rotation angles are rad; schedule times and Hamiltonian coefficients carry no documented unit. The full convention and its exceptions are in CODING_STANDARDS.md additions.
- Every numeric change ships with the before-and-after evidence REVIEW.md asks for.
</important>

## Branches and PRs

<important if="you create a branch, open or title a PR, or merge one">
Decision 2 (Qilimanjaro Software branch conventions), as this repo works today:
- Branch from `main`, named after the GitHub issue: `issue-<n>-<topic>`. The PR is linked to its issue. PR titles are a short name for the change, without the issue key. TODO(human): decide whether titles adopt Decision 2's `[TASK-ID] Name`.
- PRs are squash-merged, so the PR title becomes the commit subject on `main`.
</important>

<important if="you make a user-visible change">
- Add a towncrier fragment `changes/<PR number>.<type>.md` (`feature`, `bugfix`, `doc`, `removal` or `misc`); never edit `CHANGELOG.md` by hand. `uv run --no-sync towncrier create --no-edit <PR number>.<type>.md --content "..."` makes one; the first fragment after a release recreates the directory.
</important>

<important if="you cut a release or backport a fix">
- A release PR from `release/X.Y.Z`, titled `[Release] X.Y.Z`, bumps the version in `pyproject.toml` and `uv.lock` and builds the changelog with towncrier. The GitHub Release with the bare tag `X.Y.Z` runs `.github/workflows/publish.yml` (PyPI) and `.github/workflows/sphinx.yml` (docs). Publish then starts the `qilisdk-X.Y.Z` release line and opens the `postrelease/X.Y.Z` PR that moves `main` to the next `.dev0`.
- A fix for a release line merges to `main` first; the label `backport qilisdk-X.Y.Z` makes `.github/workflows/backport.yml` open the backport PR.
</important>

## Agent workflow
- Gardener: TODO(human) (owns the environment-change list after each retro)
- Trust rung: 1 (Pair), proposed: the human drives and reads every diff live. QiliSim is numeric code with no reference set yet.
- Scratch output (research, specs, evidence) goes to `.agents/artifacts/`, which is gitignored.

## Agent skills

<important if="you read, file, label or triage issues">
### Issue tracker
GitHub Issues, to confirm with the SDK team; older work was tracked in Linear team SDK (`SDK-###`). See `docs/agents/issue-tracker.md`.
### Triage labels
The five canonical roles, mapped onto the existing `question` and `wontfix` labels; three are still to be created. See `docs/agents/triage-labels.md`.
</important>

<important if="you name a concept or record a decision">
### Domain docs
Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
</important>

## Completion
Work is ready for review when the verify command passes and its output is attached to the PR. The verify command runs the gates of the workflows that run on pull requests.
