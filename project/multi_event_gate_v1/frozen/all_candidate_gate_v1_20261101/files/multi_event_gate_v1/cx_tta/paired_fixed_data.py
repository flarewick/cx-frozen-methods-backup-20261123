"""S2e: whole-scene omission with frozen S2a background and coordinates."""
from pathlib import Path
from collections import Counter
import numpy as np
from .paths import ROOT,new_run,write_json
from .audit import digest
from .probe_candidates import read,verify
from .matched_recovery_data import load as load_background
from .fixed_recovery_dataset import load_fixed_cache
from .recovery_context_v2 import project_context

BACKGROUND=ROOT/'outputs/effect_learning/matched_recovery_20261002_140640_079637'
CACHE=ROOT/'outputs/feature_caches/fixed_recovery_20261003_031535_304147'
CONDITIONS={'include_all':None,**{f'omit_switch_{s}':f'turn_switch:{s}' for s in (421000,421001,421007)}}


def hierarchy_weights(rows):
    """Equal task, scene within task, branch within scene, row within branch."""
    tasks={r['task'] for r in rows};counts=Counter(r['branch_group'] for r in rows)
    return np.array([1/(len(tasks)*len({q['group'] for q in rows if q['task']==r['task']})*
        len({q['branch_group'] for q in rows if q['group']==r['group']})*counts[r['branch_group']])
        for r in rows],np.float32)


def load(run):
    p=read(run/'protocol.json');verify(p)
    with np.load(run/'data.npz') as z:
        arrays={k:z[k].copy() for k in ('base','target','valid')};context=z['context'].copy()
    with np.load(run/'schedule.npz') as z:schedule={k:z[k].copy() for k in z.files}
    return p,arrays,context,schedule


def prepare():
    parent,old,old_context,old_schedule=load_background(BACKGROUND)
    for name in ('preflight_audit.json','candidate_audit.json'):
        if not read(BACKGROUND/name)['passed']:raise ValueError('Audited background required')
    inputs,labels,metadata=load_fixed_cache(CACHE,scope='developer_same_task_scene_holdout')
    cp=read(CACHE/'protocol.json')
    with np.load(parent['preprocess_source']) as z:pre={k:z[k].copy() for k in z.files}
    with np.load(cp['preprocess']) as z:
        parity={k:dict(equal=np.array_equal(pre[k],z[k]),max_abs=float(np.max(np.abs(pre[k]-z[k]))))
                for k in ('pca_components','pca_mean','context_mean','context_std')}
    with np.load(CACHE/'features.npz') as z:
        projected=project_context(z['joint'],z['state'],np.array(cp['feature_history_indices']),pre)
    context=projected[cp['feature_current_indices']]
    offset=len(parent['rows']);rows=[dict(r,source_block='background') for r in parent['rows']]
    rows += [dict(r,source_block='fixed_new') for r in metadata]
    arrays={k:np.concatenate([old[k],inputs[k] if k=='base' else labels[k]]) for k in old}
    train=list(parent['train_indices'])+list(range(offset,len(rows)))
    base_weights=np.r_[old_schedule['row_weights'][0],hierarchy_weights(metadata)]
    edges=[dict(e,branch_group=e['group'],source_block='background') for e in parent['seam_pairs']]
    new_edges=[]
    for after,before in enumerate(cp['history_indices']):
        if before<0:continue
        a,b=metadata[before],metadata[after]
        if a['branch_group']!=b['branch_group'] or a['frame']+32!=b['frame'] or not labels['valid'][before].all():
            raise ValueError('Non-contiguous or cross-branch seam')
        new_edges.append(dict(before=offset+before,after=offset+after,last=31,group=b['group'],
            task=b['task'],branch_group=b['branch_group'],source_block='fixed_new'))
    edges+=new_edges
    base_pairs=np.r_[old_schedule['pair_weights'][0],hierarchy_weights(new_edges)]
    local={i:j for j,i in enumerate(train)}
    pairs=np.array([[local[e['before']],e['last'],local[e['after']]] for e in edges],np.int64)
    ws=[];ps=[]
    for omitted in CONDITIONS.values():
        w=base_weights.copy();pw=base_pairs.copy()
        w[[i for i,r in enumerate(rows) if r['group']==omitted]]=0
        pw[[i for i,e in enumerate(edges) if e['group']==omitted]]=0
        ws.append(w);ps.append(pw)
    run=new_run('paired_fixed',group=('effect_learning',))
    np.savez_compressed(run/'data.npz',**arrays,context=np.concatenate([old_context,context]))
    np.savez_compressed(run/'schedule.npz',train_indices=np.array(train),row_weights=np.stack(ws),
        pair_indices=pairs,pair_weights=np.stack(ps))
    files=[BACKGROUND/n for n in ('protocol.json','preflight_audit.json','candidate_audit.json','schedule.npz')]
    files += [CACHE/n for n in ('protocol.json','audit.json','data.npz','features.npz','training_interface_check.json')]
    files += [run/'data.npz',run/'schedule.npz']
    files += [ROOT/'cx_tta'/n for n in ('paired_fixed_data.py','train_paired_fixed.py','audit_paired_fixed.py',
        'summarize_paired_fixed.py','recovery_context_v2.py','fixed_recovery_dataset.py')]
    files += [ROOT/'scripts/train_paired_fixed.sh']
    p=dict(status='frozen_before_training',source=str(BACKGROUND),cache=str(CACHE),rows=rows,
        background_rows=offset,new_indices=list(range(offset,len(rows))),train_indices=train,
        calibration_indices=parent['calibration_indices'],initializations=parent['initializations'],
        conditions=list(CONDITIONS),omitted_groups=CONDITIONS,config=parent['config'],
        preprocess_source=parent['preprocess_source'],preprocess_expert_source_indices=parent['preprocess_expert_source_indices'],
        preprocessing_parity=parity,seam_pairs=edges,encodings=parent['encodings'],seeds=parent['seeds'],
        updates=parent['updates'],lr=parent['lr'],parameters=parent['parameters'],
        source_sha256={**parent['source_sha256'],**{str(f):digest(f) for f in files}},
        training='Same611-row full batch/order,1400updates,AdamW lr.0003 decay.01,41742parameters. '
            'Reuse six actual S2a initial files. No sample selection or checkpoint selection. '
            'All background row and seam weights unchanged. New block total1, equal task/scene/branch/row; '
            'new seam block total1 balanced task/scene/branch/edge over branches with seams. '
            'Only target scene row/seam weights zero; no renormalization/reallocation. New shoe stays fixed.',
        preprocessing='Project34 raw frozen features using exact S2a outer expert-only PCA/scaler; '
            'arm state/pi and real32-step past. S2d coordinates differ and are not concatenated. No PCA/scaler fitting.',
        evaluation='Final1400 only; each switch scene with vs without own supervision. Both encodings, all3seeds. '
            'Raw1 and original-cap1 independently reported vs base; per-scene/per-branch/per-arm errors, '
            'explicit same-branch seams only. Original calibration is developer diagnostic, no old validation. '
            'Original recovery imitation and preservation thresholds unchanged. No best-seed or task-success claims.',
        scope='S2e same-task whole-scene developer omission diagnostic on3 switch scenes, not independent confirmation '
            'or task-success evidence. Shoe only1new scene is background. No router/TTA/newcollection/liveevaluation.')
    write_json(run/'protocol.json',p);print('RUN',run,flush=True);return run
