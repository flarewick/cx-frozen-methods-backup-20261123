"""Localize saved errors and describe target-curve complexity, offline only."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import ROOT, new_run, write_json

SOURCE = ROOT/'outputs/effect_learning/recovery_temporal_20261002_081938_551576'
ARMS = [0,1,2,3,4,5,7,8,9,10,11,12]


def read(p):
    return json.loads(p.read_text())


def numpy_time(horizon, mode):
    t = np.linspace(0, 1, horizon)
    if mode == 'repeat':
        v = np.repeat(t[:, None], 8, axis=1)
    elif mode == 'fourier':
        v = np.column_stack([f(2*np.pi*k*t) for f in (np.sin,np.cos) for k in (1,2,4,8)])
    else:
        raise ValueError('Unknown encoding')
    v -= v.mean(axis=0)
    v /= np.maximum(np.sqrt(np.mean(v*v,axis=0)),1e-6)
    return np.column_stack((t,v/np.sqrt(8)))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    source=parser.parse_args().source.resolve();p=read(source/'protocol.json');audit=read(source/'audit.json')
    if not audit['passed'] or not read(source/'temporal_audit.json')['passed']:
        raise ValueError('Completed audited S1t run required')
    if digest(source/'data.npz')!=p['data_sha256']:
        raise ValueError('Changed frozen inputs')
    with np.load(source/'data.npz') as z:
        data={k:z[k].copy() for k in z.files}
    ids=p['parts']['recovery'];records=[];curves=[]
    files=[source/'protocol.json',source/'audit.json',source/'temporal_audit.json',source/'data.npz',
           Path(__file__),ROOT/'cx_tta/time_encoding.py']
    for condition in ('baseline','combined'):
        for seed in p['seeds']:
            file=source/f'{condition}_s{seed}_u1400_recovery.npz';files.append(file)
            if digest(file)!=audit['source_sha256'][str(file)]:
                raise ValueError('Changed saved prediction')
            with np.load(file) as z:
                for i,gid in enumerate(ids):
                    valid=z['valid'][i];row=p['rows'][gid]
                    if not np.array_equal(valid,data['valid'][gid]):raise ValueError('Changed valid mask')
                    for arm,side in enumerate(('left','right')):
                        position=z['position'][i,:,arm];angle=z['angle'][i,:,arm]
                        bad=valid&((position>.01)|(angle>3));at=np.flatnonzero(bad)
                        records.append(dict(condition=condition,seed=seed,group=row['group'],frame=row['frame'],arm=side,
                            valid_count=int(valid.sum()),failed_offsets=at.tolist(),
                            native_counts=(row['frame']+at+1).tolist(),failed_count=int(bad.sum()),
                            failed_interior_count=int((bad&(np.arange(len(valid))>0)&(np.arange(len(valid))<int(valid.sum())-1)).sum()),
                            quartiles=[dict(start=j,count=int(valid[j:j+8].sum()),bad=int(bad[j:j+8].sum()),
                                mean_position_mm=None if not valid[j:j+8].any() else float(position[j:j+8][valid[j:j+8]].mean()*1000))
                                for j in range(0,32,8)]))
    for gid in ids:
        row=p['rows'][gid];valid=data['valid'][gid]
        residual=(data['target'][gid]-data['base'][gid])[:,ARMS].astype(float)
        entries={}
        for mode in ('repeat','fourier'):
            basis=np.c_[np.ones(32),numpy_time(32,mode)][valid]
            coefficients=np.linalg.lstsq(basis,residual[valid],rcond=None)[0]
            error=basis@coefficients-residual[valid]
            entries[mode]=dict(design_rank=int(np.linalg.matrix_rank(basis)),
                arm_rmse_rad=float(np.sqrt(np.mean(error**2))))
        curves.append(dict(group=row['group'],frame=row['frame'],valid_actions=int(valid.sum()),
            oracle_per_block_coefficients=True,descriptive_fits=entries))
    run=new_run('recovery_time',group=('reports','analyses'))
    write_json(run/'result.json',dict(status='complete',passed=True,source=str(source),
        localization=records,curve_description=curves,
        basis=dict(frequencies=[1,2,4,8],extra_channels=8,relative_time=True,
            scaling='Zero channel mean and RMS 1/sqrt(8) on the entire32-position grid, both modes.',
            source='Fixed powers-of-two bank; not selected by oracle fits or held scores.'),
        source_sha256={str(f):digest(f) for f in files},
        scope='Descriptive developer recovery data. Per-block least-squares coefficients use teacher labels; '
            'NOT a deployable model, shared fit, architecture lower bound, or generalization result. '
            'No teacher coefficients, phase, absolute native count or future input used by the proposed adapter.'))
    print(run,flush=True)
    print(json.dumps(dict(curves=curves,combined_bad=sum(x['failed_count'] for x in records if x['condition']=='combined'),
        combined_bad_interior=sum(x['failed_interior_count'] for x in records if x['condition']=='combined'))))


if __name__=='__main__':main()
