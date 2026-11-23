from pathlib import Path
import json
from native_seven_task_gate_expansion_v1.contract import T,P,read,mapped,sha
from native_onpolicy_recovery_v3.common import verify
O=T/'outputs/gated_postexit_recovery_v2'
B=T/'outputs/native_seven_gated_dev_v3/dev_gated'
ROOT=mapped(read(B/'execution_state.json')['root'])
def write(path,value):
 path=Path(path)
 if path.is_relative_to(ROOT):path=O/'audits'/path.relative_to(ROOT)
 assert path.resolve().is_relative_to(O.resolve());path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
