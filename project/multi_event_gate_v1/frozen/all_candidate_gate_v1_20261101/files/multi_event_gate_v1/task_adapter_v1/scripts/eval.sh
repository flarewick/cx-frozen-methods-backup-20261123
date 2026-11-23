#!/usr/bin/env bash
set -euo pipefail
TASK_ADAPTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$TASK_ADAPTER_ROOT/scripts/launch.sh" task_adapters.evaluate "$@"
