# qilisdk

[![Python Versions](https://img.shields.io/pypi/pyversions/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![PyPI Version](https://img.shields.io/pypi/v/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![License](https://img.shields.io/pypi/l/qilisdk.svg)](#license)

**qilisdk** is a Python framework for writing digital and analog quantum algorithms and executing them across multiple quantum backends. Its modular design streamlines the development process and enables easy integration with a variety of quantum platforms.

> **Note**: The instructions below focus on developing and contributing to qilisdk. For installation and usage as an end user, see the [Usage](#usage) section (placeholder).

---

## Table of Contents
- [qilisdk](#qilisdk)
  - [Table of Contents](#table-of-contents)
  - [Development](#development)
    - [Prerequisites](#prerequisites)
    - [Setup \& Dependency Management](#setup--dependency-management)
    - [Testing](#testing)
    - [Linting \& Formatting](#linting--formatting)
    - [Type Checking](#type-checking)
    - [Changelog Management](#changelog-management)
    - [Contributing](#contributing)
  - [Usage](#usage)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Development

This section covers how to set up a local development environment for qilisdk, run tests, enforce code style, manage dependencies, and contribute to the project. We use a number of tools to maintain code quality and consistency:

- **[uv](https://pypi.org/project/uv/)** for dependency management and packaging.
- **[ruff](https://beta.ruff.rs/docs/)** for linting and code formatting.
- **[mypy](http://mypy-lang.org/)** for static type checking.
- **[towncrier](https://github.com/twisted/towncrier)** for automated changelog generation.

### Prerequisites

- Python **3.10+** (we test against multiple versions, but 3.10 is the minimum for local dev).
- [Git](https://git-scm.com/) for version control.
- [uv](https://pypi.org/project/uv/) for dependency management.

### Setup & Dependency Management

1. **Clone the repository**:
   ```bash
   git clone https://github.com/qilimanjaro-tech/qilisdk.git
   cd qilisdk
   ```

2. **Install [uv](https://pypi.org/project/uv/) globally** (if not already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Sync dependencies**:
   - We maintain a `pyproject.toml` listing all dev and optional requirements.
   - To install the dev environment locally, run:
     ```bash
     uv sync
     ```
     This sets up a virtual environment and installs all pinned dependencies (including `ruff`, `mypy`, `towncrier`, etc.).
   - To install extra dependencies such as `qibo-backend`, run:
     ```bash
     uv sync --extra qibo-backend -extra ...
     ```
     This sets up a virtual environment and installs all pinned dependencies (previous), plus the specified extras.

4. **Activate the virtual environment**:
   - uv typically creates and manages its own environment, e.g., `.venv/`.
   - Run:
     ```bash
     source .venv/bin/activate
     ```
     *(Exact command can vary depending on your shell and OS.)*

Now you can run all development commands (tests, linting, etc.) within this environment.

### Testing

TODO: to_be_filled

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

### Type Checking

We use [**mypy**](http://mypy-lang.org/) for static type checking. This helps ensure our code is type-safe and maintainable.

```bash
mypy qilisdk
```

If you have extra modules or tests you want type-checked, specify them:

```bash
mypy qilisdk tests
```

*(We encourage developers to annotate new functions, classes, and methods with type hints.)*

### Changelog Management

We manage our changelog using [**towncrier**](https://github.com/twisted/towncrier). Instead of editing `CHANGELOG.md` directly, **each pull request** includes a small *news fragment* file in the `changes/` directory describing the user-facing changes.

For example, if you create a PR with id #123 adding a new feature, you add:
```
changes/123.feature.rst
```
Inside this file, you briefly describe the new feature:
```rst
Added a new `cool_feature` in the `qilisdk.extras` module.
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
   mypy qilisdk
   pytest tests
   ```
5. **Commit** and push your branch to your fork. `pre-commit` will also run the checks automatically.
6. **Open a Pull Request** against the `main` branch here.

Our CI will run tests, linting, and type checks. Please make sure your branch passes these checks before requesting a review.

---

## Usage

*(Placeholder for end users. Provide a brief snippet on how to install qilisdk and use its main features. For example:)*

```bash
pip install qilisdk
```

```python
from qilisdk import core_feature

result = core_feature.do_something()
print(result)
```

---

## License

This project is licensed under the [Apache License](LICENSE).

---

## Acknowledgments

- Thanks to all the contributors who help develop qilisdk!
- [uv](https://pypi.org/project/uv/) for making dependency management smoother.
- [ruff](https://beta.ruff.rs/docs/), [mypy](http://mypy-lang.org/), and [towncrier](https://github.com/twisted/towncrier) for their amazing tooling.

---

Feel free to open [issues](https://github.com/qilimanjaro-tech/qilisdk/issues) or [pull requests](https://github.com/qilimanjaro-tech/qilisdk/pulls) if you have questions or contributions. Happy coding!
