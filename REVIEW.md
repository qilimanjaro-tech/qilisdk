# qilisdk review contract

Shared core v0.2 (Qilimanjaro Software), a draft until the software leads agree it. Copied verbatim, followed by qilisdk additions. Applies to humans and agents. AI review is advisory: a qualified human reads every line of the same revision before merge.

## Findings
- Each finding gives `path:line` and quotes the line, then the trigger, the consequence, the evidence and a concrete next step. No citation, no finding.
- Separate confirmed defects from questions and missing evidence. Recheck earlier findings on the current revision and state the inspected scope.
- Review on two axes: standards (`CODING_STANDARDS.md`, citing the section) and spec (the linked ticket, PRD or TRD: missing, extra, wrong; quote the spec line).
- Green CI is evidence, not a verdict: say what it covers and what it misses.

## Severity
Lead with the highest.
- **blocker**: wrong results, data loss, a security hole, or a broken invariant from AGENTS.md. Fix before merge.
- **major**: a likely bug, or a design problem the next change will trip over. Fix, or justify in the PR.
- **minor**: worth fixing, not blocking.
- **nit**: taste. At most 3 per review; drop the rest.

## qilisdk additions
- **Python and C++ together.** A change to QiliSim's behavior keeps the Python side (`src/qilisdk/backends/qilisim.py`) and the C++ core (`src/qilisdk_cpp/`) in step, with tests on both sides: `tests/unit_python/` and `tests/unit_cpp/`. Green CI covers the C++ suite on Linux only.
- **Backends agree.** A change to how a functional executes is checked on every backend that runs it (QiliSim, CUDA-Q, QuTiP). The external backends run in their own CI jobs on Python 3.11 and 3.13, not under the local verify command.
- **Optional dependencies.** A backend or interop library imported eagerly, instead of through the lazy loading in `src/qilisdk/_optionals.py` (as `src/qilisdk/backends/__init__.py` does), breaks `pip install qilisdk` for users without it. Platform Builds is the only CI job that installs no extras.
- **Public API and docs.** A public API change goes through a deprecation or links an accepted decision. A new public symbol is in its package's `__all__` and `__init__.pyi` and on its page under `docs/modules/`; changed examples pass `bash scripts/docs.sh`.
- **Changelog.** A user-visible change carries a towncrier fragment (`[tool.towncrier]` in `pyproject.toml`). CI does not check for one, so the reviewer does.
- **Public repository.** No internal links, hostnames, customer names or credentials in the diff, the PR text or the commit messages.
- **Performance.** A change to a QiliSim hot path states before-and-after timings on a named input in the PR, because no CI job measures speed. TODO(human): confirm, and name the benchmark input.
- **Units.** Values follow the convention in CODING_STANDARDS.md additions; check every conversion where a value leaves this repo (compiler, driver, instrument).
- **Parity.** Any numeric change ships before-and-after output, with each difference explained: from the cross-backend checks in `tests/integration/backends/test_all_backend_integration.py` and the independent NumPy reference in `tests/integration/noise/test_non_markovian_noise_reference.py` where they reach the changed code (`uv run --no-sync pytest tests/integration`, which no CI job runs), otherwise from a synthetic input that does (for a schedule or circuit, the README examples; for a gate, its matrix).
