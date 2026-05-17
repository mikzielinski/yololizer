#!/usr/bin/env bash
# Copy static UI to docs/ for GitHub Pages
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/docs"
cp "$ROOT/frontend/index.html" "$ROOT/docs/index.html"
# docs/config.js — Pages API URL (not frontend/config.js)
touch "$ROOT/docs/.nojekyll"
echo "Built docs/ for GitHub Pages ($(wc -c < "$ROOT/docs/index.html") bytes)"
