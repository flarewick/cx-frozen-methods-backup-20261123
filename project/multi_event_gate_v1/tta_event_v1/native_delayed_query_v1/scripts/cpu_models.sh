#!/usr/bin/env bash
set -euo pipefail
PROBE="$(cd -- "$(dirname -- "$0")/.." && pwd)"
TTA="$(dirname -- "$PROBE")"
export PYTHONPATH="$PROBE:$TTA:$(dirname -- "$TTA"):$(dirname -- "$(dirname -- "$TTA")")/deployment"
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu WANDB_MODE=disabled PYTORCH_NVML_BASED_CUDA_CHECK=1
export TORCH_WARM_POOL=0
export ARROW_DEFAULT_MEMORY_POOL=system JE_ARROW_MALLOC_CONF=background_thread:false MALLOC_CONF=background_thread:false
cd -- "$PROBE"
exec /opt/cx_venv/bin/python -u -B models.py
