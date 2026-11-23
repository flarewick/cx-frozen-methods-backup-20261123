#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CX_PYTHON:-/data/cj/robo_pi/250010033/qfw/robotwin/venv_pi05_rt/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CX_GPU:-0}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME="$PROJECT/cache/xdg" HF_HOME="$PROJECT/cache/huggingface"
export TORCH_HOME="$PROJECT/cache/torch" WANDB_MODE=disabled
cd -- "$PROJECT"
exec "$PYTHON" -B -m cx_tta.train_structured_action "$@"
