"""Audited schema-3 temporal inputs; privileged targets are stored separately."""
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .dataset import context_features
from .object_effect_labels import entity_effects
from .paths import ROOT, new_run, write_json
from .short_window_batch import DEFAULT_PLAN, validate_plan

SCALE = np.array([np.pi]*6+[1.]+[np.pi]*6+[1.])


def frame_features(path):
    with np.load(path, allow_pickle=False) as data:
        vector = context_features([data[k+'_rgb'] for k in ('head','right','left')], data['command_state'])
        return np.r_[vector, data['arm_qpos'].ravel()/np.pi, data['gripper_qpos']].astype(np.float32)


def window_inputs(current, history, history_actions, base_actions, residual, caps):
    """No time-after-execution, masks, GT, future pixels, IDs or task names."""
    if base_actions.shape != (8,14) or residual.shape != (8,14) or history_actions.shape != (8,14):
        raise ValueError('Eight-action input horizon required')
    state = np.r_[current, history, (history_actions/SCALE).ravel(), (base_actions/SCALE).ravel()]
    action = (residual/caps).ravel()
    if not np.isfinite(np.r_[state,action]).all():raise ValueError('Nonfinite inputs')
    return state, action


def load_jobs():
    plan=json.loads(DEFAULT_PLAN.read_text());validate_plan(plan)
    expected={j['id']:j for j in plan['jobs']};seen=set();jobs=[]
    for task in sorted({j['task'] for j in plan['jobs']}):
        path=ROOT/'outputs/short_window_batches'/plan['batch_id']/task/'index.json'
        index=json.loads(path.read_text())
        if index['status']!='complete' or index['plan_sha256']!=digest(DEFAULT_PLAN):raise ValueError('Incomplete/changed index')
        for row in index['records']:
            if row['id'] in seen or row['id'] not in expected:raise ValueError('Duplicate/unplanned job')
            job=expected[row['id']]
            if any(row[k]!=v for k,v in job.items()):raise ValueError('Job mismatch')
            seen.add(row['id']);jobs.append(row)
    if seen!=set(expected):raise ValueError('Missing jobs')
    return jobs


def main():
    from .audit_short_windows import audit_run
    run=new_run('short_dataset',group=('effect_learning',));print(run,flush=True)
    contexts=[];actions=[];records=[];labels=[];excluded=[];sources=[]
    for job in load_jobs():
        source=Path(job['run']);audit_run(source)
        result=json.loads((source/'result.json').read_text())
        manifest=json.loads((source/'manifest.json').read_text())
        if (manifest['task'],manifest['source_seed'],manifest['chunk'])!=(job['task'],job['seed'],job['chunk']):raise ValueError('Run identity mismatch')
        caps=np.array(json.loads((source/'proposal_spec.json').read_text())['caps'])
        sources.append({'id':job['id'],'run':str(source),'result_sha256':digest(source/'result.json'),'manifest_sha256':digest(source/'manifest.json')})
        baseframes={f['offset']:f for f in result['branches']['base']['frames']}
        for name,branch in result['branches'].items():
            frames={f['offset']:f for f in branch['frames']}
            features={i:frame_features(source/f['observed_file']) for i,f in frames.items()}
            with np.load(source/f'transition_{name}.npz') as d: a=dict(d)
            base=a['proposed_actions']-a['residual_actions']
            for w in branch['windows']:
                lo,hi=w['start'],w['stop'];meta=dict(task=job['task'],group=f"{job['task']}:{job['seed']}",job=job['id'],branch=name,start=lo,stop=hi,split='train')
                if not (w['positive_time'] and w['full_stride'] and lo-8 in frames):
                    excluded.append(dict(meta,reason='partial_zero_time_or_missing_history',window=w));continue
                history_actions=a['history_actions'] if lo==0 else a['proposed_actions'][lo-8:lo]
                state,action=window_inputs(features[lo],features[lo-8],history_actions,base[lo:hi],a['residual_actions'][lo:hi],caps)
                start,end=frames[lo],frames[hi]
                label=dict(temporal=entity_effects(start['physical_offline_only']['offline_only'],end['physical_offline_only']['offline_only'],branch['articulation_limits_offline_only']),
                           visibility_before=start['visibility_offline_only'],visibility_after=end['visibility_offline_only'],duration_s=w['duration_s'])
                # Later branches are not same-state interventions; never assign them this target.
                if lo==0 and hi in baseframes and next(x for x in result['branches']['base']['windows'] if x['start']==0)['full_stride']:
                    label['counterfactual']=entity_effects(baseframes[hi]['physical_offline_only']['offline_only'],end['physical_offline_only']['offline_only'],branch['articulation_limits_offline_only'])
                    label['base_visibility_after']=baseframes[hi]['visibility_offline_only']
                contexts.append(state);actions.append(action);records.append(meta);labels.append(label)
        print('DATASET '+job['id'],flush=True)
    np.savez_compressed(run/'inputs.npz',context=np.asarray(contexts,np.float32),action=np.asarray(actions,np.float32))
    write_json(run/'labels_offline.json',labels)
    write_json(run/'index.json',dict(status='complete',records=records,excluded=excluded,sources=sources,
        inputs_sha256=digest(run/'inputs.npz'),labels_sha256=digest(run/'labels_offline.json'),plan_sha256=digest(DEFAULT_PLAN),
        scope='Train-only coarse RGB/history learning-interface data, not frozen semantic features or deployable M1'))
    print(f'COMPLETE rows={len(records)} excluded={len(excluded)} output={run}',flush=True)

if __name__=='__main__':main()
