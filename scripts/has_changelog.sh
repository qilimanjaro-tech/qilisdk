#!/bin/bash

# --------------------------------------------------------------------------
# This script checks that this branch has a changelog entry.
# --------------------------------------------------------------------------
# Time estimate: basically instant
# --------------------------------------------------------------------------

# Get the name of the branch
branch_name=$(git rev-parse --abbrev-ref HEAD)

# Only enforce on feature branches named like "sdk-XXX-description"
if [[ $branch_name =~ ^sdk-[0-9]+- ]]; then
    git fetch --quiet origin main
    base=$(git merge-base origin/main HEAD)
    files_changed=$(git diff "$base" --name-only)
    if ! echo "$files_changed" | grep -qE '^changes/.*\.md$'; then
        echo "Error: No changelog entry found for branch $branch_name. Please add a changelog entry via towncrier." >&2
        echo "Files changed in this branch:" >&2
        echo "$files_changed" >&2
        exit 1
    fi
    echo "Changelog found."
fi
