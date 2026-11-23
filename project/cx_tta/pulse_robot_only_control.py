"""Same-width control: robot state/known plan/horizon, no object or image inputs."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import new_run,write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();index=json.loads((a.dataset/'index.json').read_text())
    if digest(a.dataset/'inputs_oracle.npz')!=index['inputs_sha256'] or digest(a.dataset/'labels_offline.npz')!=index['labels_sha256']:raise ValueError('Changed source')
    with np.load(a.dataset/'inputs_oracle.npz') as z:inputs=dict(z)
    if inputs['geometry'].shape[1]!=169 or inputs['vision'].shape[1]!=6912:raise ValueError('Unknown layout')
    inputs['vision'][:]=0;inputs['geometry'][:,:84]=0
    run=new_run('pulse_robot_only_dataset',group=('effect_learning',));np.savez_compressed(run/'inputs_oracle.npz',**inputs);shutil.copyfile(a.dataset/'labels_offline.npz',run/'labels_offline.npz')
    index.update(inputs_sha256=digest(run/'inputs_oracle.npz'),labels_sha256=digest(run/'labels_offline.npz'),control_source=str(a.dataset),control_source_index_sha256=digest(a.dataset/'index.json'),scope='Same width and labels, zero visual features and first84 object/relation geometry columns; retain measured proprioception28, known base FK plan56, horizon1 and candidate residual448. No task/entity IDs as numeric input; controls robot/plan/horizon information confounded in residual-only baseline.')
    write_json(run/'index.json',index);print(run,flush=True)

if __name__=='__main__':main()
