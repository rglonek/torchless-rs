#!/bin/bash

set -euxo pipefail

# if we are in scripts, cd ..
if [ "$(basename "$(pwd)")" == "scripts" ]; then
    cd ..
fi

# check for uncommitted changes
echo "Checking for uncommitted changes"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: uncommitted changes detected. Commit or stash them first."
    exit 1
fi

# get the version from the Cargo.toml
echo "Getting version from Cargo.toml"
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "Version: $VERSION"

# check if the version is already tagged
echo "Checking if the version is already tagged"
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "Error: tag v$VERSION already exists"
    exit 1
fi

# check that local branch is up-to-date with remote
echo "Checking that local branch is up-to-date with remote"
git fetch
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse '@{u}' 2>/dev/null) || { echo "Warning: no upstream branch set, skipping remote check"; REMOTE="$LOCAL"; }
if [ "$LOCAL" != "$REMOTE" ]; then
    echo "Error: local branch is not up-to-date with remote. Pull or push first."
    exit 1
fi

# run sanity checks
echo "Running sanity checks"
bash ./scripts/sanity.sh

# tag the release
echo "Tagging release"
git tag -a "v$VERSION" -m "Release $VERSION"
echo "Pushing tag"
git push origin "v$VERSION"
echo "Done: Release tagged and pushed"
