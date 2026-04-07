#!/usr/bin/env bash
# Install GR00T-H / GR00T-H-IDM Python environment (repo root).
set -euo pipefail

# When cache and project are on different filesystems (e.g. NFS), avoid uv hardlink warnings.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: Run this script from the GR00T-H repository root (expected pyproject.toml)." >&2
  exit 1
fi

echo "uv sync (Python 3.10) — installs dependencies + this repo in editable mode into .venv/"
uv sync --python 3.10

echo ""
echo "Done. FlashAttention is NOT part of the default dependency set."
echo "On Linux + NVIDIA CUDA, install it after sync:"
echo "  uv pip install flash-attn==2.7.4.post1 --no-build-isolation"
echo "See docs/GR00T-H-IDM.md §1.3"
