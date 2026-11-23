#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Scheduler is CPU-only. Each child explicitly receives --gpu / CUDA_VISIBLE_DEVICES.
exec bash "$PROJECT/scripts/cpu.sh" -m cx_tta.frozen_event_pool "$@"
