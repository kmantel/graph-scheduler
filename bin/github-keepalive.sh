#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_IDS="$(gh api 'repos/{owner}/{repo}/actions/workflows' | jq -r '.workflows[] | .id')"
while read -r workflow; do
	gh api -X PUT 'repos/{owner}/{repo}/actions/workflows/'"${workflow}"'/enable'
done <<<"$WORKFLOW_IDS"
