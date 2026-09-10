#!/usr/bin/env bash
# Apply ci/branch-rules.json to the repo's `main` ruleset. The required status
# checks are what auto-merge waits on, so the automation PRs (roll, rc bump,
# nightly bump, goldens, release) merge on green CI and never before.
set -euo pipefail

repo="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
rules="$(dirname "$0")/branch-rules.json"
name="$(jq -r .name "$rules")"

id="$(gh api "repos/${repo}/rulesets" --jq ".[] | select(.name == \"${name}\") | .id")"
if [[ -n "${id}" ]]; then
    gh api -X PUT "repos/${repo}/rulesets/${id}" --input "$rules" > /dev/null
    echo "updated ruleset ${name} (${id}) on ${repo}"
else
    gh api -X POST "repos/${repo}/rulesets" --input "$rules" > /dev/null
    echo "created ruleset ${name} on ${repo}"
fi
gh api "repos/${repo}/rulesets/$(gh api "repos/${repo}/rulesets" --jq ".[] | select(.name == \"${name}\") | .id")" \
    --jq '.rules[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context'
