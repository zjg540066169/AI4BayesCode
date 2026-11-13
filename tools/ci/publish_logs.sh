#!/usr/bin/env bash
# Development only: push a folder of CI logs to a branch of its own, so the
# logs can be read with git without signing in to GitHub.
# Usage: GH_TOKEN=... bash tools/ci/publish_logs.sh <branch> <dir>
branch=$1; dir=$2
[ -d "$dir" ] || { echo "no $dir"; exit 0; }
tmp=$(mktemp -d)
cp -R "$dir"/. "$tmp"/
cd "$tmp" || exit 1
printf 'run %s\nsha %s\njob %s\n' "$GITHUB_RUN_ID" "$GITHUB_SHA" "$GITHUB_JOB" > RUN.txt
git init -q -b "$branch"
git config user.name github-actions
git config user.email github-actions@users.noreply.github.com
git add -A
git commit -qm "CI logs, run $GITHUB_RUN_ID"
git push -qf "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$branch"
