"""Run the already audited physical carry with current-policy admission only."""
import runpy
import sys
from policy import L, verify_rules, require_admission
from observer import admission_snapshot
sys.path.insert(0, str(L/'official_baseline_v14'))
sys.path.insert(0, str(L/'scripts'))
from host import checkpoints
import gpu_policy
import gpu_doctor

if __name__ == '__main__':
    verify_rules()
    gpu_policy.inventory = admission_snapshot
    gpu_policy.require_admission = require_admission
    gpu_doctor.verify_server_rules = verify_rules
    runpy.run_path(str(L/'scripts/continue_official_physical_v14.py'), run_name='__main__')
