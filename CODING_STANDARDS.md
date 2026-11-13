# qilisdk coding standards

Shared core v0.2 (Qilimanjaro Software), a draft until the software leads agree it. Copied verbatim, followed by qilisdk additions. Mechanical rules live in the tools configured in `pyproject.toml` and in CI; this file holds the judgement they cannot make. The review contract is `REVIEW.md`; findings cite the section they apply.

## Tests

Tests are behaviour tests: they drive the public interface at an agreed seam and assert what a caller observes.

- **Seams.** Test at the seams listed in the additions below. A new seam is named in the PR description before tests target it.
- **Observable outcomes.** Assert returned values, raised errors, emitted programs and waveforms, and results read back through the public reader.
- **Independent expected values.** Take expected values from a literal, a worked example, the specification, a published reference or a hardware-validated fixture, never recomputed the way the code computes them.
- **Boundary mocks.** Mock instrument drivers and hardware execution, network, databases, the file system when needed, time and randomness. The package's own classes and functions run for real.
- **Public reach.** When a behaviour is visible only through a `_private` name, add a public accessor or move the test to the seam above.
- **Regressions.** A bug fix starts with a test that fails on the bug.
- **Coverage is evidence.** A line counts as covered when a behaviour test exercises it. Hardware-only code goes in the coverage `omit` list with a reason.

## Code

- **Design.** Prefer deep modules: a small interface over a lot of behaviour. Name things with the terms in `GLOSSARY.md`; a new concept gets a glossary entry first.
- **Units.** Physical quantities carry their unit in the name, a type alias or the docstring (Hz, ns, V, rad, dBm), fixtures included, in the convention the additions state.
- **Suppressions.** Every suppression (`# noqa`, `# type: ignore`, `# ty: ignore`, `# pragma: no cover`, a coverage `omit`) names its rule code and a one-line reason.
- **Comments.** Explain why. A comment that excuses a workaround becomes a type, a lint or a test.
- **Dependencies.** A dependency on another Qilimanjaro package names the released version that carries the needed behaviour.
- **Compatibility.** Public API changes go through a deprecation release unless the PR links an accepted decision to break compatibility.

## Change hygiene

- User-facing changes add a towncrier fragment in `changes/`, in repos that use towncrier.
- Interface or architecture decisions link their decision record; proposals stay labelled as proposals until accepted.

## qilisdk additions

- **Seams:** the public imports the README uses (`from qilisdk.digital import Circuit`, `from qilisdk.backends import QiliSim`, and so on); `backend.execute(functional, readout)` and the result it returns (`src/qilisdk/backends/backend.py`); `QTensor` methods; a gate's matrix; a transpiler pass's output circuit. The C++ core is tested through GoogleTest in `tests/unit_cpp/`, mirroring `src/qilisdk_cpp/`. TODO(human): agree the seams.
- **Boundary mocks:** the SpeQtrum HTTP client is replaced by a stand-in for `httpx.Client` and `load_credentials` is patched (`tests/unit_python/speqtrum/test_speqtrum.py`); the keyring is patched with `monkeypatch` (`tests/unit_python/speqtrum/test_keyring.py`). CUDA-Q and QuTiP tests skip through `pytest.importorskip` when the backend is not installed. Randomness is seeded (`ExecutionConfig(seed=...)`).
- **Units:** rotation angles are rad (`src/qilisdk/digital/gates.py:953`). Schedule times (`total_time`, `dt`) and Hamiltonian coefficients carry no documented unit (`src/qilisdk/analog/schedule.py:93`); a noise channel's duration and its `t1` only need to share one (`src/qilisdk/noise/amplitude_damping.py:60`). The OpenQASM 3 importer converts `ns`, `us` and `ms` durations to seconds (`src/qilisdk/utils/openqasm/openqasm3.py:479`). Wall-clock times in results are seconds (`src/qilisdk/core/result.py:24`). TODO(human): document the time and energy convention of schedules and Hamiltonians.
- **Coverage:** no `fail_under` in `pyproject.toml`. Codecov sets the gates (`.github/codecov.yml`): the patch target is 100% and the project may drop by at most 1%, on Python and C++ coverage combined from the Python 3.11 job. `bash scripts/coverage.sh` reproduces it on Linux with `diff-cover --fail-under=99`.
- **Dependencies:** a floor raised for a specific reason (a fix, a CVE, wheels for a new Python) carries a comment with that reason, as the floors in `pyproject.toml` do, because the next upgrade needs to know whether it still matters.
- **Tooling:** the C++ style is `.clang-format` and `.clang-tidy`. A C++ or Python source file carries the Apache header naming Qilimanjaro Quantum Tech; ruff checks Python, `bash scripts/copyrights.sh` checks both.
