"""Inventory executed teacher labels and exact legal inputs without new collection."""
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import ROOT,new_run,write_json

SUMMARY=ROOT/'outputs/reports/audits/recovery_stages_20261001_092030_485008/teacher_summary.json'
HELD=ROOT/'outputs/recovery_stage_controls/place_shoe/probe_20261001_155532_413918'
BOUNDED=ROOT/'outputs/held_bounded_continuations/place_shoe/gate_20261002_035042_869420'

def read(p):return json.loads(p.read_text())

def signature_file(path):
    with np.load(path) as z:
        arrays={k:z[k].copy() for k in z.files}
    if any(k not in arrays for k in ('head_rgb','right_rgb','left_rgb','command_state_raw','command_state')):raise ValueError('Legal input schema missing')
    signature=dict(images={n:array_hash(arrays[k+'_rgb']) for n,k in [('head_camera','head'),('right_camera','right'),('left_camera','left')]},command_state=array_hash(arrays['command_state_raw']))
    return signature,arrays

def main():
    summary=read(SUMMARY);run=new_run('recovery_inventory',group=('reports','audits'));print(run,flush=True);sources=[SUMMARY,Path(__file__)];targets=run/'executed_targets';targets.mkdir();records=[]
    selected=[dict(run=str(HELD),point=4,role='verified_held_recovery',semantic='audited held-entry closure/lift/preplacement/placement/release')]
    selected.extend(dict(run=r['run'],point=r['point'],role='native_switch_recovery' if r['point']==6 else 'source_success_teacher_control' if r['point'] in (5,7) else 'restart_grasp_diagnostic',semantic='official task source; restart shoe entry is not validated held-recovery contract') for r in summary['rows'])
    # Only a completed live compiled query at an exactly matching physical/input
    # predecessor may be paired with this executed target. Never use old future L0.
    queryfolder=BOUNDED/'held_compatible';queries=[json.loads(x) for x in (queryfolder/'queries.jsonl').read_text().splitlines()];sources.extend([BOUNDED/'audit.json',queryfolder/'queries.jsonl'])
    if not read(BOUNDED/'audit.json')['passed']:raise ValueError('Query source audit failed')
    for item in selected:
        source=Path(item['run']);manifest=read(source/'manifest.json');result=read(source/'result.json');audit=read(source/'audit.json')
        if not audit['passed'] or not result['teacher_success'] or not result['native_terminal']:raise ValueError('Completed native teacher required')
        for name in ('manifest.json','result.json','audit.json','fork.json','trace.jsonl','native_actions.npz'):sources.append(source/name)
        if digest(source/'trace.jsonl')!=result['trace_sha256']:raise ValueError('Changed execution trace')
        trace=[json.loads(x) for x in (source/'trace.jsonl').read_text().splitlines()];fork=read(source/'fork.json');prefix=result['prefix_actions'];n=result['teacher_actions']
        with np.load(source/'native_actions.npz') as z:actions=z['actions'].copy()
        if len(trace)!=n or actions.shape!=(n,14) or result['action_count']!=prefix+n or [r['action_count'] for r in trace]!=list(range(prefix+1,prefix+n+1)):raise ValueError('Executed label accounting mismatch')
        if not np.array_equal(actions,np.array([r['action'] for r in trace],np.float32)):raise ValueError('Teacher actions disagree')
        plans={}
        for planid in sorted({r['plan'] for r in trace}):
            path=source/f'plan_{planid:03d}.npz';sources.append(path)
            with np.load(path) as z:plans[planid]=z['dense_commands'].copy()
        for row,action in zip(trace,actions):
            if not np.array_equal(action,plans[row['plan']][row['dense_index']].astype(np.float32)):raise ValueError('Actual action not exported source plan')
        # Include every saved image for provenance, but only 32-action decision
        # boundaries become candidate cached adapter inputs.
        observations=[]
        for path in [source/'fork.npz',*sorted(source.glob('observation_*.npz')),source/'terminal.npz']:
            offset=0 if path.name=='fork.npz' else n if path.name=='terminal.npz' else int(path.stem.split('_')[1]);expected=fork['observation'] if offset==0 else trace[offset-1]['after_observation'];actual,_=signature_file(path)
            if actual!=expected:raise ValueError('Saved observation not measured state '+str(path))
            observations.append(dict(file=str(path),offset=offset,native_count=prefix+offset,decision_boundary=offset<n and offset%32==0));sources.append(path)
        sourceconfig=Path(manifest['source'])/'resolved_config.json';sources.append(sourceconfig)
        for offset in range(0,n,32):
            path=source/'fork.npz' if offset==0 else source/f'observation_{offset:04d}.npz'
            if not path.exists():raise ValueError('Missing scheduled 32-action image')
            sig,obs=signature_file(path);physical=fork['diagnostics']['physical'] if offset==0 else trace[offset-1]['offline_only']['physical'];count=min(32,n-offset);target=np.zeros((32,14),np.float32);target[:count]=actions[offset:offset+count];valid=np.arange(32)<count
            filename=f"p{item['point']}_{'held' if source==HELD else 'native'}_n{prefix+offset}.npz";targetfile=targets/filename;np.savez_compressed(targetfile,target=target,valid=valid,source_native_counts=np.where(valid,prefix+offset+1+np.arange(32),-1),plan_ids=np.array([trace[offset+j]['plan'] if j<count else -1 for j in range(32)]))
            matches=[]
            if source==HELD:
                for q in queries:
                    if q['native_count']==prefix+offset and q['observation']==sig and q['physical']==physical:
                        basefile=queryfolder/f"base_{q['query']:04d}.npz"
                        with np.load(basefile) as z:base=z['actions']
                        if array_hash(base)!=q['base_sha256']:raise ValueError('Matched base hash mismatch')
                        matches.append(dict(file=str(basefile),sha256=digest(basefile),query=q['query'],rng=q['rng'],source='S1k held_compatible same physical/input predecessor; no future trajectory reuse'));sources.append(basefile)
            row=dict(task='place_shoe' if 'place_shoe' in str(source) else 'turn_switch',scene_seed=manifest['scene_seed'],group=f"{('place_shoe' if 'place_shoe' in str(source) else 'turn_switch')}:{manifest['scene_seed']}",role=item['role'],semantic=item['semantic'],source=str(source),source_point=item['point'],instruction=manifest['instruction'],native_count=prefix+offset,native_limit=result['native_limit'],observation=str(path),observation_sha256=digest(path),observation_signature=sig,target=str(targetfile),target_sha256=digest(targetfile),valid_actions=count,plan_ids=sorted({trace[offset+j]['plan'] for j in range(count)}),past_observation=None if offset==0 else str(source/('fork.npz' if offset==32 else f'observation_{offset-32:04d}.npz')),matched_compiled_base=matches,
                missing=['frozen joint-prefix feature']+([] if matches else ['same-state official compiled base block with recorded sampler provenance']),
                status='executed_target_and_legal_observation_verified_not_training_ready',training_approved=False)
            records.append(row)
        write_json(run/(f"p{item['point']}_{'held' if source==HELD else 'native'}_source.json"),dict(source=str(source),role=item['role'],observations=observations,plans=len(plans),prefix_actions=prefix,native_count=result['action_count'],native_limit=result['native_limit'],teacher_success=True))
    eligible=[r for r in records if r['role'] in ('verified_held_recovery','native_switch_recovery')]
    write_json(run/'result.json',dict(status='complete',passed=True,records=records,summary=dict(source_runs=5,all_candidate_chunks=len(records),candidate_recovery_chunks=len(eligible),candidate_recovery_groups=len({r['group'] for r in eligible}),same_state_live_compiled_base_chunks=sum(bool(r['matched_compiled_base']) for r in eligible),all_scheduled_images_available=True,usable_training_rows=0),
        next_steps=['No new simulator collection is needed for these saved 32-action decision images/actual teacher chunks. A bounded offline cache can generate missing joint-prefix features and compiled base samples directly from exact saved legal observations.',
            'Cache must retain frozen preprocessing/checkpoint, instruction, per-state RNG provenance, and same-state matching. After teacher divergence do not use later original L0 or bounded-branch actions as base targets.',
            'Only two original developer failed scene groups support candidate recovery labels. They can test interface/fit capacity, not independent held-out recovery or generalization.',
            'Native restart shoe diagnostic and success-source teachers are inventoried but not automatically approved as correct held-recovery/optimal or preservation labels. Current label inventory is not a training dataset.'],
        excluded=dict(run=str(BOUNDED/'compatible'),reason='Official success during unfinished restart grasp/contact loss; do not assign clean completed-stage recovery labels'),source_sha256={str(f):digest(f) for f in sorted(set(sources))},scope='CPU provenance inventory only: actual executed commands, exact legal observations and budgets audited; no new inference, collection, model training or recovery success claim. Privileged phases stay diagnostic.'))
    print(json.dumps(dict(passed=True,all_chunks=len(records),recovery_chunks=len(eligible),recovery_groups=len({r['group'] for r in eligible}),compiled_base_matches=sum(bool(r['matched_compiled_base']) for r in eligible),training_ready=0),indent=2))

if __name__=='__main__':main()
