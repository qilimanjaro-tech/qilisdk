# Issue tracker: GitHub Issues

TODO(human): confirm the tracker with the SDK team. This file records what the repository shows today; it is not yet an agreed rule.

## What the repository shows

- **GitHub Issues** is where current work is filed and picked up. `.github/ISSUE_TEMPLATE/` holds three forms, titled `[Bug]: `, `[Feature]: ` and `[Task]: `, which set the GitHub issue type. Most issues carry an area label (`core`, `digital`, `analog`, `qilisim`, `backends` and the rest of the "Touches ..." labels).
- Since August 2026 most PRs branch as `issue-<n>-<topic>` from the issue they resolve and are linked to that issue. PR titles are a short name for the change, without the issue key.
- Before that, work was tracked in **Linear team SDK**: branches `sdk-###-<topic>` and PR titles `[SDK-###] Name`. Some open issues still name their Linear ticket in the body, and `scripts/has_changelog.sh` still recognizes only `sdk-###-` branches.
- Maintenance PRs use other names: `release/X.Y.Z`, `postrelease/X.Y.Z`, `ci/<topic>` and Dependabot's `dependabot/...`.

## Conventions

- **Read an issue:** `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues:** `gh issue list --state open --json number,title,labels` with `--label` filters.
- **Read a Linear ticket** (older work, `SDK-###`): use the Linear tools (for example the Linear MCP server's `get_issue`) and refer to it by identifier.
- **Create, comment on, label or close** an issue only after the human has seen the title and body and said yes. The repository is public: an issue body carries no internal links, hostnames or customer names.
- **New issues** follow the matching form in `.github/ISSUE_TEMPLATE/`: the title prefix, the sections the form asks for, and one area label.
- **Branches and PRs:** see `## Branches and PRs` in `AGENTS.md`.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

## When a skill says "publish to the issue tracker"

Create a GitHub issue from the matching template, after the human has approved its title and body.

## When a skill says "fetch the relevant ticket"

A `#<n>` reference is a GitHub issue or PR (`gh issue view <n>`, falling back to `gh pr view <n>`); an `SDK-###` reference is a Linear ticket.
