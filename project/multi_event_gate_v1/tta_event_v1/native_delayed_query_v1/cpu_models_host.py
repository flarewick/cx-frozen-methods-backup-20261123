"""CPU-only one-thread two-model preload in the already verified container."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'))
import user_container as container
spec=importlib.util.spec_from_file_location('existing_GPU0_CPU_placement',T/'native_current_v4/cpu_placement.py')
placement=importlib.util.module_from_spec(spec);spec.loader.exec_module(placement)
plan=json.loads((T/'outputs/native_current_v4/cpu_placement_plan.json').read_text())
placement.apply(plan)
canonical=container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/C.relative_to(P)
raise SystemExit(subprocess.run(container.command(['bash',str(canonical/'scripts/cpu_models.sh')],mount_project=True)).returncode)
