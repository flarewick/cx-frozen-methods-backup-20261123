#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$HERE/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export PYTHONPATH="$PROJECT/multi_event_gate_v1/task_adapter_v1:$PROJECT/multi_event_gate_v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu
cd -- "$PROJECT/multi_event_gate_v1"
exec /opt/cx_venv/bin/python -u -B "$@"
