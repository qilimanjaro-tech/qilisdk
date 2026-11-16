# Contributing to QiliSDK

We welcome contributions! Here’s the workflow:

1. **Fork** this repository and create a feature branch.
2. **Write** your changes (code, docs, or tests).
3. **Add a news fragment** in `changes/` describing the user-facing impact.
4. **Run** the following checks locally:
   ```bash
   bash scripts/checks.sh
   ```
5. **Commit** and push your branch to your fork.
6. **Open a Pull Request** against the `main` branch here.

Our CI will run tests, linting, and type checks. Please make sure your branch passes these checks before requesting a review.

To set up a development environment, run the tests and write news fragments, see the [Development Guide](README.md#development-guide) in the README.

## AI Tools

AI tools may be used, but every PR must be opened and owned by a human who understands and can explain the change and answers review comments. PRs created autonomously by AI agents or bots, without a responsible human, will be closed.

## Code of Conduct

Everyone taking part in this project is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
