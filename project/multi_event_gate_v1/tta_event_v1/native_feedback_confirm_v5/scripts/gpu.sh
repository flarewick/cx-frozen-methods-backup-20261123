#!/usr/bin/env bash
set -euo pipefail
PROBE="$(cd -- "$(dirname -- "$0")/.." && pwd)"
TTA="$(dirname -- "$PROBE")"
export PYTHONPATH="$PROBE:$TTA/native_nextchunk_v1:$TTA/controlled_effect_v3:$TTA/controlled_effect_v2:$TTA/actuation_trace_v1:$TTA/response_factor_v5:$TTA/response_factor_v4:$TTA/response_factor_v3:$TTA/response_factor_v2:$TTA/time_align_v1:$TTA/local_effect_v1:$TTA/time_span_v1:$TTA/response_shared_v1:$TTA/readout_interface_v2:$TTA/direction_probe_v1:$TTA:$(dirname -- "$TTA"):$(dirname -- "$(dirname -- "$TTA")")/deployment"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu WANDB_MODE=disabled
export PYTORCH_NVML_BASED_CUDA_CHECK=1 TORCH_WARM_POOL=0 JE_ARROW_MALLOC_CONF=background_thread:false MALLOC_CONF=background_thread:false ARROW_DEFAULT_MEMORY_POOL=system
cd -- "$PROBE"
exec /opt/cx_venv/bin/python -u -B -m cx_tta_feedback_confirm5.preload "$@"
