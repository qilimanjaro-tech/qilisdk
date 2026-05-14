# QiliSDK

[![Python Versions](https://img.shields.io/pypi/pyversions/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![PyPI Version](https://img.shields.io/pypi/v/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![Code Coverage](https://codecov.io/gh/qilimanjaro-tech/qilisdk/graph/badge.svg?token=iUMp8nKCqJ)](https://codecov.io/gh/qilimanjaro-tech/qilisdk)
[![License](https://img.shields.io/pypi/l/qilisdk.svg)](#license)
[![Docs](https://img.shields.io/badge/docs-latest-pink.svg)](https://qilimanjaro-tech.github.io/qilisdk/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18468117.svg)](https://doi.org/10.5281/zenodo.18468117)

**QiliSDK** is an open-source Python framework for designing and executing **analog, digital, and hybrid quantum algorithms**. Its modular structure unifies circuit-based and Hamiltonian-based workflows within a single API. It provides high-level abstractions for gates, circuits, Hamiltonians, and more, while remaining fully backend-agnostic allowing a seamless switch between CPU, GPU, or QPU execution. Fast CPU simulation can be done locally using **QiliSim**, our quantum simulator written in C++.

## Installation

QiliSDK is available via [PyPI](https://pypi.org/project/qilisdk/):

```bash
pip install qilisdk
```

For other installaton options, see the [docs](https://qilimanjaro-tech.github.io/qilisdk).

## Usage

Here are just a few examples to get you started, for tutorials and full documentation, please see the [docs](https://qilimanjaro-tech.github.io/qilisdk).

### Digital Circuits

To create a simple quantum circuit:

```python
from qilisdk.digital import Circuit, H, RX, CNOT

circuit = Circuit(2)             # Create a circuit with 2 qubits
circuit.add(H(0))                # Apply Hadamard on qubit 0
circuit.add(RX(1, theta=3.14))   # Apply RX rotation on qubit 1
circuit.add(CNOT(0, 1))          # Add a CNOT gate between qubit 0 and 1
```

### Analog Evolution

To create a linear interpolation between an initial and final Hamiltonian:

```python
from qilisdk.analog import Schedule, X, Z

initial_hamiltonian = - X(0) - X(1)
final_hamiltonian = Z(0) + Z(1) + 0.5 * Z(0) * Z(1)

schedule = Schedule.linear(initial_hamiltonian, final_hamiltonian, total_time=10.0, dt=0.5)
```

## Development Guide

This section covers how to set up a local development environment for qilisdk, run tests, enforce code style, manage dependencies, and contribute to the project. We use a number of tools to maintain code quality and consistency:

- **[uv](https://pypi.org/project/uv/)** for dependency management and packaging.
- **[ruff](https://docs.astral.sh/ruff/)** for linting and code formatting.
- **[ty](https://docs.astral.sh/ty/)** for language server and static type checking.
- **[towncrier](https://github.com/twisted/towncrier)** for automated changelog generation.

### Setup & Dependency Management

For instructions on how to compile from source and set up the development environment, see the [docs](https://qilimanjaro-tech.github.io/qilisdk/main/getting_started/installation#compiling-from-source). Following those instructions will setup a virtual environment (venv) in which you can run all other development tools (e.g. tests, linting).

### Testing

We use **pytest** for the test suite. Once you have your venv set up, run the Python unit tests using:

```bash
pytest tests/unit_python
```

To run the integration tests (bigger tests which use the full QiliSDK stack and might take a bit longer):
```bash
pytest tests/integration
```

To run the CUDA, qutip, and SpeQtrum backend tests, install the optional extras first (e.g., `uv sync --all-groups --extra all-cu13`).

To run the C++ tests, you first need to recompile with C++ testing enabled:
```bash
uv sync --reinstall -Ccmake.define.tests=ON
```
The C++ testing suite can then be ran using:
```bash
./tests/unit_cpp/test_cpp
```

### Linting & Formatting

We enforce code style and best practices using [**ruff**](https://beta.ruff.rs/docs/). ruff handles:

- Lint checks (similar to flake8, pylint).
- Formatting (similar to black or isort).
- Automated fixes for certain issues.

To check linting:

```bash
ruff check
```

To automatically fix lint issues (where possible):

```bash
ruff check --fix
```

To automatically format your code:

```bash
ruff format
```

*(We recommend running `ruff check --fix` and `ruff format` before committing any changes.)*

To check the C++ code, compile with the tidy flag:
```bash
uv pip install -v -e ./ -Ccmake.build-type=Debug -Ccmake.define.tidy=ON
```
This will run clang-tidy as well as a number of C++ compiler flags for debugging. For this you will need clang-tidy installed, which can be done on Debian/Ubuntu with:
```bash
sudo apt-get install clang-tidy
```
It may also throw an error about not being able to find `omp.h`, if so, try:
```bash
sudo apt-get install libomp-dev
```
For ease of use there are also a number of bash scripts in the "scripts" folder, for instance to generate coverage reports or to run all pre-commit checks.

### Type Checking

We use [**ty**](https://docs.astral.sh/ty/) for static type checking. This helps ensure our code is type-safe and maintainable.

```bash
ty check
```

*(We encourage developers to annotate new functions, classes, and methods with type hints.)*

### Helpful Scripts

For ease of use there are also a number of scripts in the `scripts/` directory. Each of these generates a .log file with the same name as the script (`checks.sh` generates `checks.log` and so on).

To run all pre-commit checks (e.g. linting/tests):
```bash
bash scripts/checks.sh
```

To check all of the code blocks in the documentaton:
```bash
bash scripts/docs.sh
```

To generate a full coverage report of all code, run:
```bash
bash scripts/cov.sh
```
This will generate a html file which you can open in your browser to see which lines (in both the C++ and Python) are covered by the tests. We aim for 100% coverage!

### Changelog Management

We manage our changelog using [**towncrier**](https://github.com/twisted/towncrier). Instead of editing `CHANGELOG.md` directly, **each pull request** includes a small *news fragment* file in the `changes/` directory describing the user-facing changes.

For example, if you create a PR with id #123 adding a new feature, you add:
```
changes/123.feature.rst
```
Inside this file, you briefly describe the new feature:
```rst
Added a new `cool_feature` in the `qilisdk.backends` module.
```
Instead of manually creating the file, you can run:
```bash
towncrier create --no-edit
```
When we cut a new release, we update the version in `pyproject.toml` file and run:
```bash
towncrier
```
This aggregates all the news fragments into the `CHANGELOG.md` under the new version and removes the used fragments.

### Contributing

We welcome contributions! Here’s the workflow:

1. **Fork** this repository and create a feature branch.
2. **Write** your changes (code, docs, or tests).
3. **Add a news fragment** (if applicable) in `changes/` describing the user-facing impact.
4. **Run** the following checks locally:
   ```bash
   ruff check --fix
   ruff format
   ty check
   pytest tests
   ```
5. **Commit** and push your branch to your fork. `pre-commit` will also run the checks automatically.
6. **Open a Pull Request** against the `main` branch here.

Our CI will run tests, linting, and type checks. Please make sure your branch passes these checks before requesting a review.

## License

This project is licensed under the [Apache License](LICENSE).

## Acknowledgments

- Thanks to all the contributors who help develop qilisdk!
- [uv](https://docs.astral.sh/uv/) for making dependency management smoother.
- [ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/), and [towncrier](https://github.com/twisted/towncrier) for their amazing tooling.

---

Feel free to open [issues](https://github.com/qilimanjaro-tech/qilisdk/issues) or [pull requests](https://github.com/qilimanjaro-tech/qilisdk/pulls) if you have questions or contributions. Happy quantum coding!
