"""Fixed full-batch recovery inclusion experiment; reuse audited outer coordinates."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT, new_run, write_json
from .audit import digest
from .probe_candidates import read, verify
from .train_semantic_pilot import balanced_weights
from .train_rotation_curriculum import state_hash
from .time_encoding import TimeEncodedAdapter

SOURCE=ROOT/'outputs/effect_learning/scene_crossfit_20261002_114334_518712'
CONDITIONS={'both':None,'omit_shoe':'place_shoe:320006','omit_switch':'turn_switch:320002'}


def load(run):
    p=read(run/'protocol.json');verify(p)
    ids=p['source_indices']
    with np.load(p['data_source']) as z:data={k:z[k][ids].copy() for k in ('base','target','valid')}
    with np.load(p['context_source']) as z:context=z['context'][ids].copy()
    with np.load(run/'schedule.npz') as z:schedule={k:z[k].copy() for k in z.files}
    return p,data,context,schedule


def prepare():
    parent=read(SOURCE/'protocol.json');verify(parent)
    for name in ('preflight_audit.json','candidate_audit.json'):
        if not read(SOURCE/name)['passed']:raise ValueError('Audited source required')
    job=next(j for j in parent['jobs'] if j['name']=='outer')
    ids=sorted(job['train']+parent['parts']['calibration'])
    if set(ids)&set(parent['parts']['validation']):raise ValueError('No old validation permitted')
    inverse={old:new for new,old in enumerate(ids)}
    rows=[dict(parent['rows'][i],s1y_index=i) for i in ids]
    train=[inverse[i] for i in job['train']];cal=[inverse[i] for i in parent['parts']['calibration']]
    with np.load(job['context']) as z:weights=z['weights'][ids].copy()
    edges=parent['seam_pairs'];local={i:j for j,i in enumerate(train)}
    pairs=np.array([[local[inverse[e['before']]],e['last'],local[inverse[e['after']]]] for e in edges],np.int64)
    pair_weights=balanced_weights(edges).numpy().astype(np.float32)
    all_weights=[];all_pairs=[]
    for condition,omitted in CONDITIONS.items():
        w=weights.copy();pw=pair_weights.copy()
        if omitted is not None:
            w[[i for i,r in enumerate(rows) if r['group']==omitted]]=0
            pw[[i for i,e in enumerate(edges) if e['group']==omitted]]=0
        all_weights.append(w);all_pairs.append(pw)
    run=new_run('matched_recovery',group=('effect_learning',))
    np.savez_compressed(run/'schedule.npz',train_indices=np.array(train),row_weights=np.stack(all_weights),
        pair_indices=pairs,pair_weights=np.stack(all_pairs))
    initial=[]
    for encoding in parent['encodings']:
        for seed in parent['seeds']:
            torch.manual_seed(seed);m=TimeEncodedAdapter(157,encoding=encoding,envelope=True)
            f=run/f'initial_{encoding}_s{seed}.pt';torch.save(m.state_dict(),f)
            initial.append(dict(encoding=encoding,seed=seed,path=str(f),sha256=digest(f),state_sha256=state_hash(m)))
    files=[SOURCE/f for f in ('protocol.json','result.json','preflight_audit.json','candidate_audit.json','data.npz')]
    files += [run/'schedule.npz',Path(job['context']),Path(job['preprocess'])]+[Path(x['path']) for x in initial]
    files += [ROOT/'cx_tta'/f for f in ('matched_recovery_data.py','train_matched_recovery.py','audit_matched_recovery.py')]
    files += [ROOT/'scripts/train_matched_recovery.sh']
    protocol=dict(status='frozen_before_training',source=str(SOURCE),rows=rows,source_indices=ids,
        train_indices=train,calibration_indices=cal,initializations=initial,conditions=list(CONDITIONS),
        omitted_groups=CONDITIONS,config=parent['config'],data_source=str(SOURCE/'data.npz'),
        context_source=job['context'],preprocess_source=job['preprocess'],preprocess_expert_source_indices=job['preprocess_fit'],
        encodings=parent['encodings'],seeds=parent['seeds'],updates=parent['updates'],lr=parent['lr'],
        parameters=parent['parameters'],seam_pairs=[dict(e,before=inverse[e['before']],after=inverse[e['after']]) for e in edges],
        source_sha256={**parent['source_sha256'],**{str(f):digest(f) for f in files}},
        training='Same full588-row batch and order on EVERY one of1400updates; AdamW lr.0003/default decay.01; '
                 'same actual initial state file for all three conditions. Same model, float32, deterministic settings, '
                 'native BC+SE3+within-block and boundary differences. No minibatch sampling, renormalization or reallocation. '
                 'Only omitted-scene row loss weights and its seam weights set to zero. Forward is row-independent MLP.',
        preprocessing='Exact existing outer PCA/scaler/context bytes; experts572/32scenes only fitted them; no refit.',
        evaluation='Final1400 only. Same two recovery scenes compared with versus without their own supervision; '
                   'report both scenes separately, other scene transfer, expert-fit and anchor-fit side effects, '
                   'original8expert+1cup calibration developer scenes. Do not use old S1y validation. '
                   'Raw and original-cap candidate, teacher-known component choices including continuous sequence.',
        scope='S2a paired development dependence diagnostic, two recovery scenes only; not a new independent '
              'benchmark or task-success result. Exclusion tests supervision dependence, not its unique cause '
              '(memory/data scarcity/representation/distribution shift remain possible). No router/TTA/new collection.',
        source_limitations='Old evaluations directory missing before cleanup; frozen derivative arrays/config/model sources '
                           'required for this offline experiment all still exist and match hashes. No raw replay claim.')
    write_json(run/'protocol.json',protocol)
    print('RUN',run,flush=True)
    print('COUNTS',dict(Counter(r['kind'] for r in rows)),flush=True)
    return run
