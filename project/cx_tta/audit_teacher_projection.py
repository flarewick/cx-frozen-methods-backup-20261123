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
            elif folder.name in ('projected','projected32'):expected=base+np.clip(desired-base,-caps,caps)
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
                measured=np.asarray(meta['measured_arm_qpos'])
                checks['plans'] &= np.allclose(measured,physical['arm_qpos'],rtol=0,atol=1e-7)
                first_plan_seen.add(pid)
            if previous_teacher is not None and row['phase']!=previous_teacher['phase']:
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
        checks['planning_attempts'] &= bool(failures)==(result['teacher_exit']=='motion_plan_failed')
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


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    manifest=json.loads((run/'manifest.json').read_text())
    result=json.loads((run/'result.json').read_text())
    protocol=json.loads(Path(manifest['protocol']).read_text())
    expected=protocol['points'][manifest['point']]['observation_signature']
    with np.load(run/'common_base.npz',allow_pickle=False) as z:common=z['base'].copy()
    common_rng=json.loads((run/'common_rng.json').read_text())
    branches=[audit_branch(run/b,manifest,expected,common,common_rng) for b in manifest['branches']]
    records={r['branch']:r for r in result['reports']}
    full_prefix=[json.loads(l) for l in (run/'teacher_full/trace.jsonl').read_text().splitlines()][:32]
    short_prefix=[json.loads(l) for l in (run/'teacher32/trace.jsonl').read_text().splitlines()][:32]
    prefix_equal=(len(full_prefix)==len(short_prefix)==32 and all(
        a['action']==b['action'] and a['desired']==b['desired'] and
        a['after_observation']==b['after_observation'] and
        a['offline_only']['physical']==b['offline_only']['physical']
        for a,b in zip(full_prefix,short_prefix)))
    checks=dict(code_unchanged=all(digest(Path(p))==h for p,h in manifest['code_sha256'].items()),
        protocol_unchanged=digest(Path(manifest['protocol']))==manifest['protocol_sha256'],
        complete=result['status']=='complete' and set(records)==set(manifest['branches']),
        base_sham=records['base']['trajectory_sha256']==records['sham_zero']['trajectory_sha256'],
        unbounded_teacher=records['teacher_full']['success'] and not records['teacher_full']['l0_fallback'],
        teacher32_same_prefix=prefix_equal,
        branches=all(b['passed'] for b in branches),paired_rng=True)
    if 'projected32' in records:
        long=[json.loads(l) for l in (run/'projected/trace.jsonl').read_text().splitlines()][:32]
        short=[json.loads(l) for l in (run/'projected32/trace.jsonl').read_text().splitlines()][:32]
        checks['projected32_same_prefix']=(len(long)==len(short)==32 and all(
            a['action']==b['action'] and a['desired']==b['desired'] and
            a['after_observation']==b['after_observation'] and
            a['offline_only']['physical']==b['offline_only']['physical'] for a,b in zip(long,short)))
    reference={}
    for branch in branches:
        for i,rng in enumerate(branch.pop('queries_rng')):
            if i in reference:checks['paired_rng'] &= reference[i]==rng
            else:reference[i]=rng
    report=dict(passed=all(checks.values()),checks=checks,branches=branches,
        auditor_sha256=digest(Path(__file__)),scope='One-state privileged feasibility; no trained adapter or generalization result')
    write_json(run/'audit.json',report);print(json.dumps(report,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
