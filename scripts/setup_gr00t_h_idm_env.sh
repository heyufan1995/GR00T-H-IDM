#!/usr/bin/env bash
# Install GR00T-H / GR00T-H-IDM Python environment (repo root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: Run this script from the GR00T-H repository root (expected pyproject.toml)." >&2
  exit 1
fi

echo "[1/2] uv sync (Python 3.10)"
uv sync --python 3.10

echo "[2/2] Editable install"
uv pip install -e .

echo "Done. Optional: uv pip install flash-attn==2.7.4.post1 --no-build-isolation"
echo "Docs: docs/GR00T-H-IDM.md"
