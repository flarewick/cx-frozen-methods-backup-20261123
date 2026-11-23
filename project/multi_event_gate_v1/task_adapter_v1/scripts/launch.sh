#!/usr/bin/env bash
set -euo pipefail
TASK_ADAPTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd -- "$TASK_ADAPTER_ROOT/.." && pwd)"
RT=/data/cj/robo_pi/250010033/qfw/robotwin
PYTHON="${CX_PYTHON:-$RT/venv_pi05_rt/bin/python}"
MODULE="$1"; shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_ADAPTER_ROOT:$WORKSPACE" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu
export PATH="$(dirname -- "$PYTHON"):$RT/bin:$PATH"
GPU=0; PREVIOUS=''; CPU=0
for arg in "$@"; do
  if [[ "$PREVIOUS" == --gpu ]]; then GPU="$arg"; fi
  if [[ "$arg" == --gpu=* ]]; then GPU="${arg#--gpu=}"; fi
  if [[ "$arg" == --dry-run || "$arg" == --help || "$arg" == -h ]]; then CPU=1; fi
  if [[ "$PREVIOUS" == --phase && ( "$arg" == prepare || "$arg" == status || "$arg" == audit ) ]]; then CPU=1; fi
  PREVIOUS="$arg"
done
cd -- "$WORKSPACE"
if [[ "$CPU" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=''
else
  export CUDA_VISIBLE_DEVICES="$GPU"
  source "$RT/setup_user_nvidia_graphics.env"
  export CUDA_HOME="$RT/cuda_unified" LD_LIBRARY_PATH="$RT/cuda_unified/lib64:${LD_LIBRARY_PATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi
exec "$PYTHON" -B -m "$MODULE" "$@"
