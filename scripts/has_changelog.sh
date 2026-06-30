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
    # Make sure we have an up-to-date reference to the remote main, so the
    # diff does not depend on a possibly-stale local "main" ref.
    git fetch --quiet origin main

    # Compare against the merge-base with origin/main so we only look at what
    # this branch introduced (committed + working-tree changes), regardless of
    # how far the local refs have drifted.
    base=$(git merge-base origin/main HEAD)

    if ! git diff "$base" --name-only | grep -qE '^changes/.*\.md$'; then
        echo "Error: No changelog entry found for branch $branch_name. Please add a changelog entry via towncrier." >&2
        exit 1
    fi

    echo "Changelog found."
fi
