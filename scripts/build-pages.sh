#!/usr/bin/env bash
# Copy static UI to docs/ for GitHub Pages
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/docs"
cp "$ROOT/frontend/index.html" "$ROOT/docs/index.html"
cp "$ROOT/docs/config.js" "$ROOT/docs/config.js"
touch "$ROOT/docs/.nojekyll"
echo "Built docs/ for GitHub Pages ($(wc -c < "$ROOT/docs/index.html") bytes)"
