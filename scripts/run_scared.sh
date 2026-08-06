#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src"

exec python -u -m endo3reval \
  --config "${PROJECT_ROOT}/configs/scared.local.json" \
  "$@"
