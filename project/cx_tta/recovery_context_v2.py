"""Repair recovery preprocessing using the original training state/pi contract.

Old paired observations/features/actions remain immutable. Never train directly
from the v1 recovery context: it appended raw arm state instead of state/pi.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .dataset import context_features,read_observation
from .paths import ROOT,new_run,write_json,replace_json
from .recovery_dataset import load_recovery_cache,SCOPE

SOURCE=ROOT/'outputs/feature_caches/recovery_20261002_072911_916687'
FIT=ROOT/'outputs/effect_learning/fit_budget_20261002_070634_070111'
ARMS=[0,1,2,3,4,5,7,8,9,10,11,12]

def read(p):return json.loads(p.read_text())

def project_context(features,raw_state,history,pre):
    state=np.asarray(raw_state,np.float32).copy();state[:,ARMS]/=np.pi
    # Match sklearn PCA.transform: center after projection, in float32.
    visual=features.astype(np.float32)@pre['pca_components'].T
    visual-=pre['pca_mean'].reshape(1,-1)@pre['pca_components'].T
    current=np.c_[visual,state];past=np.zeros_like(current);has=history>=0
    for i,j in enumerate(history):
        if j>=i or j< -1:raise ValueError('Noncausal history')
    past[has]=current[history[has]]
    return ((np.c_[current,past,has.astype(np.float32)]-pre['context_mean'])/pre['context_std']).astype(np.float32)

def prepare(source):
    inputs,labels,metadata=load_recovery_cache(source,scope=SCOPE)
    p=read(source/'protocol.json');oldfit=read(FIT/'protocol.json')
    if not read(FIT/'audit.json')['passed']:raise ValueError('Fit audit required')
    with np.load(FIT/'data.npz') as z:original={k:z[k].copy() for k in z.files}
    pres={}
    for name in ('semantic','coarse'):
        with np.load(FIT/f'{name}_preprocess.npz') as z:pres[name]={k:z[k].copy() for k in z.files}
    features_source=Path(p['feature_cache']);encoded=read(features_source/'result.json')['records']
    golden=[];source_files=[source/'protocol.json',source/'audit.json',source/'data.npz',source/'features.npz',
        FIT/'protocol.json',FIT/'audit.json',FIT/'data.npz',FIT/'semantic_preprocess.npz',FIT/'coarse_preprocess.npz',
        Path(__file__),ROOT/'cx_tta/dataset.py',features_source/'result.json']
    for task in sorted({r['source']['task'] for r in encoded}):
        record=next(r for r in encoded if r['source']['kind']=='expert' and r['source']['task']==task)
        src=record['source'];source_files.extend([Path(record['file']),Path(src['source'])])
        if digest(Path(record['file']))!=record['sha256'] or digest(Path(src['source']))!=src['source_sha256']:
            raise ValueError('Changed golden source')
        with np.load(record['file']) as z:frames=z['frames'][:2];joint=z['joint'][:2].astype(np.float32)
        raw=[];coarse=[]
        with h5py.File(src['source'],'r') as h:
            for frame in frames:
                images,state=read_observation(h,int(frame));raw.append(state);coarse.append(context_features(images,state)[14:])
        ids=np.array([next(i for i,r in enumerate(oldfit['rows']) if r['group']==src['group'] and r['frame']==frame) for frame in frames])
        if len(ids)!=2 or pres['semantic']['history_indices'][ids[0]]!=-1 or pres['semantic']['history_indices'][ids[1]]!=ids[0]:
            raise ValueError('Golden history mismatch')
        for name,features in [('semantic',joint),('coarse',np.stack(coarse))]:
            actual=project_context(features,np.stack(raw),np.array([-1,0]),pres[name]);expected=original[name][ids]
            max_error=float(np.abs(actual-expected).max())
            state_error=float(np.abs(actual[:,64:78]-expected[:,64:78]).max())
            if max_error>2e-4 or state_error>2e-6:raise ValueError(f'Training-preprocessing parity failed {name}: {max_error}, {state_error}')
            golden.append(dict(task=task,representation=name,indices=ids.tolist(),max_context_error=max_error,max_state_error=state_error))
    with np.load(source/'features.npz') as z:joint=z['joint'].copy();raw=z['state'].copy()
    coarse=[]
    for row in p['rows']:
        with np.load(row['observation']) as z:
            coarse.append(context_features([z[k+'_rgb'] for k in ('head','right','left')],z['command_state'])[14:])
        source_files.append(Path(row['observation']))
    history=np.array(p['history_indices']);contexts={name:project_context(v,raw,history,pres[name])
        for name,v in [('semantic',joint),('coarse',np.stack(coarse))]}
    old=inputs['context'];pre=pres['semantic'];old_unscaled=old*pre['context_std']+pre['context_mean']
    if not np.allclose(old_unscaled[:,64:78],raw,atol=1e-6):raise ValueError('Unexpected v1 bug signature')
    if np.allclose(old,contexts['semantic']):raise ValueError('Expected raw-state preprocessing discrepancy absent')
    run=new_run('recovery_context',group=('feature_caches',));print(run,flush=True)
    np.savez_compressed(run/'data.npz',**contexts,base=inputs['base'],**labels,history_indices=history)
    rows=[dict(**r,pilot_split='recovery_fit',frame=r['native_count'],kind='recovery') for r in metadata]
    report=dict(status='complete',passed=True,schema='recovery_context_v2_state_div_pi',rows=rows,source=str(source),
        fit_preprocessing=str(FIT),history_indices=history.tolist(),data_sha256=digest(run/'data.npz'),
        source_sha256={str(f):digest(f) for f in sorted(set(source_files))},golden_training_checks=golden,
        correction=dict(arm_state='raw native command / pi; grippers unchanged; same scaling for history',
            pca='Match sklearn centering-after-projection convention; no PCA refit',
            max_context_change=float(np.abs(old-contexts['semantic']).max()),
            affected='Only v1 derived context. RGB, joint-prefix features, raw states, compiled base and actual teacher labels unchanged. No recovery model was trained from v1.'),
        scope=SCOPE)
    write_json(run/'audit.json',report)
    write_json(source/'context_contract_erratum.json',dict(status='v1_context_superseded_do_not_train_directly',
        explanation='S1r raw arm-state vs original training state/pi mismatch; prior audit mirrored the same wrong construction.',
        replacement=str(run),replacement_audit_sha256=digest(run/'audit.json'),raw_data_and_base_valid=True))
    replace_json(source/'progress.json',dict(phase='context_superseded_use_v2',replacement=str(run),raw_data_and_base_valid=True))
    print('PREPARED',run,'max corrected context difference',report['correction']['max_context_change'],flush=True)

def load_context(run):
    run=Path(run);audit=read(run/'audit.json')
    if not audit['passed'] or audit['schema']!='recovery_context_v2_state_div_pi':raise ValueError('Corrected audited context required')
    for f,sha in audit['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed context source '+f)
    if digest(run/'data.npz')!=audit['data_sha256']:raise ValueError('Changed context data')
    with np.load(run/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    return audit,data

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args();threadpool_limits(2);prepare(args.source.resolve())

if __name__=='__main__':main()
