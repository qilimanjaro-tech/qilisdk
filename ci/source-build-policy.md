# Dependency build policy

Fixed CI and developer installs use the committed lock and deny third-party
source builds. CI pins the verified uv 0.13.8. Developer scripts need uv 0.13.8
or newer. The library's own native/editable build, CMake Debug/Release/coverage
settings, dependency versions, Python/platform matrices and lowest-direct test
jobs are unchanged. Cache reads are disabled on the hardened install paths so
source-built cached wheels do not stand in for published wheels.

## One existing documentation source exception

The docs group already locks `sphinx-multiversion==0.2.5` to Git commit
`cd4c0db66473f04d7e037188f5f13444574b4b0e` from
<https://github.com/sphinx-contrib/multiversion>. There is no registry wheel in
the lock for this Git-sourced selection. Docs/all-group installs explicitly
allow only that package to build using `--no-binary-package sphinx-multiversion`
alongside `--no-build`. Other third-party packages remain wheel-only.

`scripts/check_source_build_policy.py` checks the exact version, repository,
branch and immutable commit before those installs. Updating that selection
requires reviewing its source and build prerequisites and changing the check
in the same PR. Do not broaden the exception to fix missing wheels elsewhere.

The selected source declares `hatchling` and `hatch-requirements-txt` as
build requirements, reads its version from the package initializer, and reads
its sole runtime requirement (`sphinx>=2.1`) from requirements.txt. No source
selection or dependency version is changed by this policy.

This exception still executes the selected third-party build backend. Its
isolated build requirements are not fully locked by the application lock.
First-party native build tools also remain a separate trust boundary. This
policy is not a sandbox or a claim that no third-party code executes.

All checks use local fixtures or package builds. Publishing, GitHub Pages,
CUDA/GPU execution, hardware and production operations require their separate
human-owned workflows and approval.
