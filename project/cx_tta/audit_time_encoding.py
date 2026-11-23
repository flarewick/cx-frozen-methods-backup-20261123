"""Audit the time basis, checkpoint allocation, and actual recovery predictions."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .audit_time_encoding_dense import main as audit_dense
from .time_encoding import TimeEncodedAdapter,time_features
from .probe_recovery_time import numpy_time


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();torch.set_num_threads(2);threadpool_limits(2)
    p=read(run/'protocol.json');r=read(run/'result.json');check=read(run/'loss_interface_check.json')
    if not check['passed'] or len(check['checks'])!=4:raise ValueError('All time/decoder interfaces must pass')
    oldargv=sys.argv;sys.argv=[oldargv[0],str(run)]
    try:audit_dense()
    finally:sys.argv=oldargv
    with np.load(run/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    source=Path(p['reference_source']);prior=read(source/'protocol.json')
    if p['seam_pairs']!=prior['seam_pairs'] or p['seam_weights']!=prior['seam_weights']:
        raise ValueError('Changed temporal supervision')
    if p['rows']!=prior['rows'] or p['parts']!=prior['parts']:
        raise ValueError('Changed labels/partitions')
    with np.load(source/'weights.npz') as old,np.load(run/'weights.npz') as new:
        if any(not np.array_equal(new[c],old['baseline']) for c in p['conditions']):
            raise ValueError('Changed per-component exposure')
    basis_records=[]
    for mode in ('repeat','fourier'):
        basis=time_features(32,mode,device='cpu',dtype=torch.float64).numpy()
        expected=numpy_time(32,mode);error=float(np.abs(basis-expected).max())
        if error>1e-12:raise ValueError('Relative-time convention mismatch')
        rank=int(np.linalg.matrix_rank(np.c_[np.ones(32),basis]))
        if rank!=({'repeat':2,'fourier':10}[mode]):raise ValueError('Unexpected basis rank')
        basis_records.append(dict(mode=mode,max_numpy_error=error,design_rank_with_intercept=rank,
            extra_channel_mean=basis[:,1:].mean(0).tolist(),extra_channel_rms=np.sqrt((basis[:,1:]**2).mean(0)).tolist()))
    max_replay=0.;records=[];envelope_files=0
    for report in r['reports']:
        setting=p['conditions'][report['condition']]
        model=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=setting['envelope'])
        model.load_state_dict(torch.load(report['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        if sum(v.numel() for v in model.parameters())!=41742 or model.trunk[0].in_features!=180:
            raise ValueError('Actual checkpoint architecture mismatch')
        extra_norm=float(model.trunk[0].weight[:,-8:].detach().norm())
        if extra_norm<=0:raise ValueError('Time columns did not learn')
        for entry in report['files']:
            with np.load(entry['path']) as z:pred=z['predicted'].copy();ids=z['indices'].copy()
            if setting['envelope']:
                base=data['base'][ids][...,[6,13]];grip=pred[...,[6,13]]
                if ((grip<np.minimum(base,0))|(grip>np.maximum(base,1))).any():raise ValueError('Additional gripper overshoot')
                envelope_files+=1
            if entry['part']=='recovery':
                with torch.no_grad():
                    context=torch.as_tensor(data['semantic'][ids]);base=torch.as_tensor(data['base'][ids])
                    replay=model(context,base).numpy()
                    if not torch.equal(model(context,base,enabled=False),base):raise ValueError('Trained disabled adapter changed base')
                err=float(np.abs(replay-pred).max());max_replay=max(max_replay,err)
                if err>2e-5:raise ValueError('CPU checkpoint prediction does not match saved GPU prediction')
                records.append(dict(condition=report['condition'],seed=report['seed'],update=report['update'],
                    cpu_saved_prediction_max_error=err,extra_time_weight_norm=extra_norm))
    if len(records)!=36 or envelope_files!=66:raise ValueError('Incomplete time/decoder audit coverage')
    write_json(run/'time_encoding_audit.json',dict(passed=True,checkpoint_replays=len(records),
        gripper_envelope_files=envelope_files,max_cpu_gpu_prediction_error=max_replay,basis_checks=basis_records,
        records=records,source_sha256={str(f):digest(f) for f in
            [run/'audit.json',run/'protocol.json',run/'loss_interface_check.json',Path(__file__)]},
        scope='Independent numerical and provenance checks; CPU/GPU replay tolerance2e-5 in native commands. '
            'No independent scenes, deployment or success claim.'))
    print(json.dumps(dict(time_encoding_audit_passed=True,checkpoints=len(records),
        envelope_files=envelope_files,max_cpu_gpu_prediction_error=max_replay)))


if __name__=='__main__':main()
