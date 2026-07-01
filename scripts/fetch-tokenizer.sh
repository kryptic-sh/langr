#!/usr/bin/env bash
# Fetch a permissive LLM tokenizer vocab (Qwen3, Apache-2.0) for langr.
# The tokenizer.json is data, not committed to this repo (see .gitignore).
set -euo pipefail

REPO="${1:-Qwen/Qwen3-0.6B}"
OUT="${2:-tokenizer.json}"
URL="https://huggingface.co/${REPO}/resolve/main/tokenizer.json"

echo "fetching ${URL} -> ${OUT}"
curl -sSL -o "${OUT}" "${URL}"
echo "done ($(du -h "${OUT}" | cut -f1))"
