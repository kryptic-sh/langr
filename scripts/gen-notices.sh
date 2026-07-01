#!/usr/bin/env bash
# Generate THIRD-PARTY-LICENSES.md from the dependency tree.
# Requires cargo-about: cargo install cargo-about
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cargo-about >/dev/null 2>&1; then
    echo "cargo-about not found; install with: cargo install cargo-about" >&2
    exit 1
fi

cargo about generate about.hbs -o THIRD-PARTY-LICENSES.md
echo "wrote THIRD-PARTY-LICENSES.md"
