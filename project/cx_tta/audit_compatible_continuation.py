"""Independent saved-data audit for the S1b teacher projection gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash, json_hash
from .paths import write_json


def saved_signature(path):
    with np.load(path,allow_pickle=False) as z:
        return dict(images={k:array_hash(z[n]) for k,n in
            (('head_camera','head_rgb'),('right_camera','right_rgb'),('left_camera','left_rgb'))},
            command_state=array_hash(z['command_state_raw']))


def audit_branch(folder,manifest,expected_fork,common,common_rng):
    result=json.loads((folder/'result.json').read_text())
    fork=json.loads((folder/'fork.json').read_text())
    rows=[json.loads(l) for l in (folder/'trace.jsonl').read_text().splitlines()]
    queries=[json.loads(l) for l in (folder/'queries.jsonl').read_text().splitlines()]
    caps=np.array(manifest['caps'],np.float32)
    checks=dict(source_hashes=digest(folder/'trace.jsonl')==result['trace_sha256'] and
        digest(folder/'queries.jsonl')==result['queries_sha256'],
        fork=fork['observation']==expected_fork==saved_signature(folder/'fork.npz') and fork['physical']['matched'],
        queries=True,chain=True,actions=True,plans=True,phase_advance=True,terminal=True)
    arrays={}
    for i,q in enumerate(queries):
        with np.load(folder/f'base_{i:04d}.npz',allow_pickle=False) as z:arrays[i]=z['actions'].copy()
        checks['queries'] &= (q['query']==i and q['base_sha256']==array_hash(arrays[i]) and
            q['observation']==saved_signature(folder/f'observation_{i:04d}.npz') and
            q['rng']['before']==(queries[i-1]['rng']['after'] if i else common_rng['before']['sha256']))
    checks['queries'] &= np.array_equal(arrays[0],common)
    checks['queries'] &= queries[0]['rng']['after']==common_rng['after']['sha256']
    count=manifest['prefix_actions'];previous=expected_fork
    physical=fork['diagnostics']['physical'];stopped=False;teacher_count=0;over_cap=0
    trajectory=hashlib.sha256();plans={};first_plan_seen=set();previous_teacher=None
    for i,row in enumerate(rows):
        q,slot=divmod(i,32)
        base=np.asarray(row['base'],np.float32);action=np.asarray(row['action'],np.float32)
        if slot==0:
            checks['queries'] &= queries[q]['native_count']==count and queries[q]['observation']==previous
            checks['queries'] &= queries[q]['physical']==physical
        expected_advance=0 if stopped else 1
        checks['chain'] &= (row['query']==q and row['slot']==slot and row['advanced']==expected_advance
            and row['count']==count+expected_advance and row['before_observation']==previous
            and row['padding']==stopped and row['physics_steps']>=(1 if expected_advance else 0))
        if stopped:checks['chain'] &= row['physics_steps']==0
        checks['actions'] &= np.array_equal(base,arrays[q][slot])
        desired=row['desired']
        if desired is None:
            expected=base
        else:
            teacher_count+=1;desired=np.asarray(desired,np.float32)
            over_cap+=int((np.abs(desired-base)>caps+1e-6).sum())
            if folder.name=='sham_zero':expected=base
            elif folder.name in ('projected','projected32','compatible'):expected=base+np.clip(desired-base,-caps,caps)
            else:expected=desired
            if folder.name in ('teacher32','projected32'):checks['actions'] &= teacher_count<=32
            pid=row['plan']
            if pid not in plans:
                meta=json.loads((folder/f'plan_{pid:04d}.json').read_text())
                with np.load(folder/f'plan_{pid:04d}.npz',allow_pickle=False) as z:
                    plans[pid]=(meta,z['dense_commands'].copy(),z['indices'].copy())
            meta,dense,indices=plans[pid]
            checks['plans'] &= (meta['query']==q and row['dense_index'] in indices and
                row['phase']==meta['phase'] and np.array_equal(desired,dense[row['dense_index']]))
            if pid not in first_plan_seen:
                checks['plans'] &= meta['native_count']==count and meta['physical']==physical and meta['observation']==previous
                if folder.name!='compatible':
                    measured=np.asarray(meta['measured_arm_qpos'])
                    checks['plans'] &= np.allclose(measured,physical['arm_qpos'],rtol=0,atol=1e-7)
                first_plan_seen.add(pid)
            if folder.name!='compatible' and previous_teacher is not None and row['phase']!=previous_teacher['phase']:
                old_meta,old_dense,old_indices=plans[previous_teacher['plan']]
                offset=0 if old_meta['side']=='left' else 7
                active=slice(offset,offset+6) if old_meta['kind']=='move' else slice(offset+6,offset+7)
                err=np.abs(np.asarray(previous_teacher['action'])[active]-old_dense[-1,active]).max()
                checks['phase_advance'] &= (row['phase']==previous_teacher['phase']+1 and
                    previous_teacher['dense_index']==old_indices[-1] and err<=2.1e-6)
            previous_teacher=row
        checks['actions'] &= np.array_equal(action,expected) and np.isfinite(action).all()
        count=row['count'];previous=row['after_observation'];physical=row['offline_only']['physical']
        if stopped:checks['terminal'] &= bool(row['success'])==bool(rows[i-1]['success'])
        stopped=bool(row['success']) or count==manifest['native_limit']
        comparable={k:row[k] for k in ('query','slot','action','advanced','count','padding','success','after_observation')}
        comparable['physical']=physical;trajectory.update(json_hash(comparable).encode())
    checks['terminal'] &= (stopped and count==result['action_count'] and count<=manifest['native_limit'] and
        rows[-1]['success']==result['success'] and result['complete'] and
        result['final_diagnostics']['physical']==physical and saved_signature(folder/'terminal.npz')==previous)
    checks['accounting']=(len(rows)==32*len(queries)==result['observation_updates'] and
        len(queries)==result['queries'] and sum(r['advanced'] for r in rows)==result['continuation_actions'] and
        result['continuation_actions']==count-manifest['prefix_actions'] and teacher_count==result['teacher_actions'] and
        sum(r['physics_steps'] for r in rows)==result['physics_steps'] and
        trajectory.hexdigest()==result['trajectory_sha256'])
    if 'planning_attempts' in result:
        path=folder/'planning_attempts.jsonl'
        attempts=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        checks['planning_attempts']=(len(attempts)==result['planning_attempts'] and
            (digest(path) if path.exists() else None)==result['planning_attempts_sha256'])
        states={manifest['prefix_actions']:(fork['diagnostics']['physical'],expected_fork)}
        states.update({r['count']:(r['offline_only']['physical'],r['after_observation'])
                       for r in rows if r['advanced']})
        failures=[]
        for i,a in enumerate(attempts):
            state=states.get(a['native_count'])
            checks['planning_attempts'] &= (a['attempt']==i and state==(a['physical'],a['observation']) and
                a['offline_only']['physical']==a['physical'] and
                queries[a['query']]['native_count']<=a['native_count']<queries[a['query']]['native_count']+32)
            if not a['returned_plan'] or not a['plan_success']:
                failures.append(a)
                checks['planning_attempts'] &= saved_signature(folder/f'planning_failure_{i:04d}.npz')==a['observation']
        checks['planning_attempts'] &= bool(failures)==('planning_failed' in result['teacher_exit'] or result['teacher_exit']=='motion_plan_failed')
        checks['planning_attempts'] &= len(failures)<=1
        if failures:
            checks['planning_attempts'] &= failures[0]['native_count']==manifest['prefix_actions']+teacher_count
    if result.get('prefix_source') is not None:
        source=json.loads((folder/'prefix_source.json').read_text())
        parent=folder.parent/source['branch']
        checks['prefix_source']=(source==result['prefix_source'] and source['actions']==32 and
            source['branch']==('teacher_full' if folder.name=='teacher32' else 'projected') and
            digest(parent/'trace.jsonl')==source['trace_sha256'] and
            digest(parent/'result.json')==source['result_sha256'] and teacher_count==32 and
            result['teacher_exit']=='teacher_horizon' and result['planning_attempts']==0)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(branch=folder.name,passed=all(checks.values()),checks=checks,
        success=result['success'],teacher_actions=teacher_count,outside_cap_desired_entries=over_cap,
        result_sha256=digest(folder/'result.json'),queries_rng=[q['rng'] for q in queries])



def measurement_audit(folder):
    from .constraint_teacher import quaternion_degrees,hold_compatible
    measurements=[json.loads(x) for x in (folder/'measurements.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
    checks=dict(arrival=True,compatibility=True,phase_advance=True,plan_mode=True,immediate_transition=True,measured_joint_source=True)
    fork=json.loads((folder/'fork.json').read_text())
    states={fork['prefix_actions']:fork['diagnostics']['physical']}
    states.update({r['count']:r['offline_only']['physical'] for r in rows if r['advanced']})
    for m in measurements:
        physical=states.get(m['native_count']);arm=0 if m['side']=='left' else 1
        checks['measured_joint_source'] &= physical is not None
        if m['kind']!='move':
            checks['measured_joint_source'] &= bool(m['finger_qpos']) and physical is not None and abs(m['finger_qpos'][0]-physical['gripper_qpos'][arm])<1e-7
            continue
        c=m['constraint']
        if c is not None and physical is not None:
            checks['measured_joint_source'] &= np.allclose(np.asarray(c['planner_start_qpos']).reshape(6),physical['arm_qpos'][arm],rtol=0,atol=5.2e-6) and c['measured_native_ee_pose']==m['arrival']['pose']
        a=m['arrival'];distance=float(np.linalg.norm(np.asarray(a['pose'][:3])-m['goal'][:3]))
        angle=quaternion_degrees(a['pose'][3:],m['goal'][3:])
        checks['arrival'] &= abs(distance-a['distance_m'])<1e-9 and abs(angle-a['angle_degrees'])<1e-6 and a['reached']==bool(distance<=.005 and angle<=3.)
        c=m['constraint']
        if c is not None:checks['compatibility'] &= c['compatible']==(not c['active'] or hold_compatible(c['rotation_metric'],c['position_error_m'],c['weights'],c['project_to_goal_frame']))
    by_phase={}
    for m in measurements:by_phase.setdefault(m['phase'],[]).append(m)
    result=json.loads((folder/'result.json').read_text())
    for phase in range(result['completed_phases']):
        ms=by_phase.get(phase,[])
        if not ms:checks['phase_advance']=False;continue
        last=ms[-1]
        if last['kind']=='move':checks['phase_advance'] &= last['arrival']['reached'] and (last['constraint'] is None or last['constraint']['compatible'])
        else:
            candidates=[r for r in rows if r.get('phase')==phase and r['desired'] is not None]
            if not candidates:checks['phase_advance']=False;continue
            r=candidates[-1];meta=json.loads((folder/f"plan_{r['plan']:04d}.json").read_text());offset=6 if meta['side']=='left' else 13
            checks['phase_advance'] &= abs(r['action'][offset]-last['target'])<=2.1e-6 and bool(last['finger_qpos'])
    teacher_rows=[r for r in rows if r['desired'] is not None]
    for i,r in enumerate(teacher_rows):
        meta=json.loads((folder/f"plan_{r['plan']:04d}.json").read_text())
        if meta['kind']!='move':continue
        pre=[m for m in measurements if m['phase']==r['phase'] and m['stage']=='before_plan' and m['native_count']==meta['native_count']]
        checks['plan_mode'] &= bool(pre)
        if pre:
            c=pre[-1]['constraint'];incompatible=c is not None and not c['compatible']
            checks['plan_mode'] &= (meta['mode']=='alignment')==incompatible
        after=[m for m in measurements if m['phase']==r['phase'] and m['stage']=='after_action' and m['native_count']==r['count']]
        if not after:
            checks['immediate_transition'] &= bool(r['success'] or r['count']==500)
        elif i+1<len(teacher_rows):
            c=after[-1]['constraint'];compatible=c is None or c['compatible'];n=teacher_rows[i+1]
            if n['phase']==r['phase'] and ((r['mode']=='alignment' and compatible) or (r['mode']=='original' and not compatible)):
                checks['immediate_transition'] &= n['plan']!=r['plan']
    return {k:bool(v) for k,v in checks.items()}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);args=p.parse_args();run=args.run.resolve()
    m=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
    point=json.loads(Path(m['protocol']).read_text())['points'][m['point']]
    with np.load(run/'common_base.npz',allow_pickle=False) as z:common=z['base'].copy()
    rng=json.loads((run/'common_rng.json').read_text())
    branches=[audit_branch(run/b,m,point['observation_signature'],common,rng) for b in m['branches']]
    measured=measurement_audit(run/'compatible')
    records={r['branch']:r for r in result['reports']}
    checks=dict(code=all(digest(Path(p))==v for p,v in m['code_sha256'].items()),protocol=digest(Path(m['protocol']))==m['protocol_sha256'],complete=result['status']=='complete',branches=all(b['passed'] for b in branches),measurements=all(measured.values()),base_sham=records['base']['trajectory_sha256']==records['sham_zero']['trajectory_sha256'])
    checks['native_sources']=all(digest(Path(x['source']))==x['sha256']==digest(Path(x['snapshot'])) for x in json.loads((run/'native_sources.json').read_text()))
    report=dict(passed=all(checks.values()),checks=checks,branches=branches,measurement_checks=measured,scope='one developer checkpoint; privileged teacher full continuation, no student/generalization claim')
    write_json(run/'audit.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
