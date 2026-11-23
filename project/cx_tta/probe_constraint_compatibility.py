"""S1f: replay first capped compatible state and test the unchanged native approach."""
import argparse
import importlib
import json
import signal
import traceback
from pathlib import Path
import numpy as np
from .audit import digest
from .collect_recovery import DEFAULT_PROTOCOL, signature
from .constraint_teacher import inspect_native_constraint, alignment_goal
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
from .paths import ROOT, new_run, write_json, replace_json
from .probe_action_replay import load_source, source_runtime, source_scene_index, prime_warp_cache
from .probe_recovery_teacher import diagnostics
from .native_teacher import SceneStepGuard, export_commands
from .teacher_projection import CAPS, planner_status

PARENT=ROOT/'outputs/recovery_projections/place_shoe/gate_20261001_122755_227228'
BRANCHES=('align_capped',)
ALIGN_PARENT=ROOT/'outputs/constraint_interfaces/place_shoe/probe_20261001_133100_983784'

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not args.gpu.isdigit():p.error('GPU index required')
    alignment_audit=json.loads((ALIGN_PARENT/'audit.json').read_text())
    if not alignment_audit['passed']:raise ValueError('Alignment parent audit failed')
    alignment_row=json.loads((ALIGN_PARENT/'align_capped/trace.jsonl').read_text().splitlines()[0])
    parent=PARENT/'projected';audit=json.loads((PARENT/'audit.json').read_text())
    if not audit['passed']:raise ValueError('Parent audit failed')
    trace=[json.loads(x) for x in (parent/'trace.jsonl').read_text().splitlines()]
    attempt=[json.loads(x) for x in (parent/'planning_attempts.jsonl').read_text().splitlines()][-1]
    result=json.loads((parent/'result.json').read_text())
    for name,key in [('trace.jsonl','trace_sha256'),('planning_attempts.jsonl','planning_attempts_sha256'),('queries.jsonl','queries_sha256')]:
        if digest(parent/name)!=result[key]:raise ValueError('Parent source hash mismatch')
    parent_manifest=json.loads((PARENT/'manifest.json').read_text())
    if digest(DEFAULT_PROTOCOL)!=parent_manifest['protocol_sha256']:raise ValueError('Protocol changed')
    point=json.loads(DEFAULT_PROTOCOL.read_text())['points'][4]
    source=Path(point['source']);manifest,episode,rows=load_source(source,point['seed'])
    for name,key in [('manifest.json','source_manifest_sha256'),('resolved_config.json','source_config_sha256'),(f"traces/r0_seed{point['seed']}.jsonl",'source_trace_sha256')]:
        if digest(source/name)!=point[key]:raise ValueError('L0 source changed')
    with np.load(parent/'base_0001.npz',allow_pickle=False) as z:base=z['actions'][20:32].copy()
    if attempt['native_count']!=116 or len(base)!=12:raise ValueError('Wrong probe window')
    for i,r in enumerate(trace[52:64]):
        if r['query']!=1 or r['slot']!=20+i or not np.array_equal(base[i],np.asarray(r['action'],np.float32)):raise ValueError('Not the issued query tail')
    runtime=source_runtime(source);config=json.loads((source/'resolved_config.json').read_text())
    job=dict(parent=str(PARENT),source=str(source),point=4,scene_seed=point['seed'],branches=BRANCHES,
        start_count=116,stop_count=128,native_limit=500,remaining_query_slots=list(range(20,32)),
        caps=CAPS.tolist(),planner_stride=config['save_freq'],video=False,
        scope='privileged first compatible-state query; one extra action, no approach execution or task-success claim',
        transition='explicit orientation alignment at fixed measured native EE-reference position; original approach constraint retained',
        readiness='original native hold metric compatible only; anchor drift is diagnostic, not an added position constraint',
        planning='native robot planner, bypass only task wrapper plan_success bookkeeping; never reset task flag',
        source_sha256={str(f):digest(f) for f in [PARENT/'audit.json',PARENT/'manifest.json',parent/'result.json',parent/'trace.jsonl',parent/'queries.jsonl',parent/'base_0001.npz',parent/'planning_attempts.jsonl',DEFAULT_PROTOCOL,ALIGN_PARENT/'audit.json',ALIGN_PARENT/'align_capped/trace.jsonl',ALIGN_PARENT/'align_capped/alignment_commands.npz']})
    if args.dry_run:print(json.dumps(job,indent=2));return
    run=new_run('compatibility',group=('constraint_interfaces',point['task']));print(run,flush=True)
    files=[Path(__file__),ROOT/'cx_tta/constraint_teacher.py',ROOT/'cx_tta/native_teacher.py',ROOT/'cx_tta/teacher_projection.py',runtime/'envs/robot/robot.py',runtime/'envs/robot/planner.py']
    write_json(run/'manifest.json',dict(**job,code_sha256={str(f):digest(f) for f in files}))
    task=guard=None;reports=[]
    def timeout(*_):raise TimeoutError('Interface probe exceeded 1200 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    try:
        prime_warp_cache(source,run);configure_runtime(run,runtime,resolve_tokenizer())
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.set_num_threads(1)
        cls=getattr(importlib.import_module('envs.'+point['task']),point['task'])
        config.update(save_path=str(run/'runtime_data'),cx_runtime=str(runtime),save_data=False,eval_video_log=False,eval_video_save_dir=None)
        side=attempt['side'];goal=attempt['goal']['pose'];constraint=attempt['goal']['constraint']
        for branch in BRANCHES:
            folder=run/branch;folder.mkdir();replace_json(run/'progress.json',dict(branch=branch,phase='replaying',reports=reports))
            task=cls();seed_all(point['seed']);task.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']),seed=point['seed'],**dict(config,need_plan=True,is_test=True))
            task.set_instruction(instruction=point['instruction']);obs=task.get_obs()
            guard=SceneStepGuard(task.scene);task.scene=guard
            def execute(action):
                nonlocal obs
                n=task.take_action_cnt
                if n>=128 or task.eval_success or task.step_lim!=500:raise RuntimeError('Probe budget/terminal changed')
                guard.in_native=True
                try:task.take_action(np.asarray(action,np.float32))
                finally:guard.in_native=False
                obs=task.get_obs()
                if task.take_action_cnt!=n+1:raise RuntimeError('Action did not advance once')
            for c in range(point['chunk']):
                if signature(obs)!=rows[c]['observation_components']:raise RuntimeError('L0 prefix mismatch')
                for a in rows[c]['base_executed']:execute(a)
            if signature(obs)!=point['observation_signature'] or state_snapshot(task)!=json.loads((parent/'fork.json').read_text())['diagnostics']['physical']:
                raise RuntimeError('Initial fork mismatch')
            for r in trace[:52]:
                if signature(obs)!=r['before_observation']:raise RuntimeError('Teacher prefix predecessor mismatch')
                execute(r['action'])
                if signature(obs)!=r['after_observation'] or state_snapshot(task)!=r['offline_only']['physical']:raise RuntimeError('Teacher prefix replay mismatch')
            if task.take_action_cnt!=116 or state_snapshot(task)!=attempt['physical'] or signature(obs)!=attempt['observation']:raise RuntimeError('Rejection state mismatch')
            def pure_call(call):
                before=(state_snapshot(task),task.take_action_cnt,guard.steps,task.plan_success)
                value=call()
                if before!=(state_snapshot(task),task.take_action_cnt,guard.steps,task.plan_success):raise RuntimeError('Planning/inspection changed physical state, budget or task flag')
                return value
            def inspect():return pure_call(lambda:inspect_native_constraint(task.robot,side,goal,constraint))
            def plan(pose,mask,label):
                with planner_status(task,side) as status:
                    value=pure_call(lambda:getattr(task.robot,side+'_plan_path')(pose,constraint_pose=mask))
                write_json(folder/(label+'.json'),dict(pose=pose,constraint=mask,status=value['status'],motion_gen=status,native_count=int(task.take_action_cnt)))
                return value
            initial=inspect();anchor=np.asarray(initial['measured_native_ee_pose'][:3])
            write_json(folder/'fork.json',dict(native_count=116,inspection=initial,diagnostics=diagnostics(task),observation=signature(obs),exact_source_replay=True))
            reason='window_end';executed=0;accepted=None
            if branch=='rejection':
                value=plan(goal,constraint,'original_rejection');reason='original_query_rejected' if value['status']!='Success' else 'unexpected_original_acceptance'
                if initial['compatible'] or value['status']=='Success':raise RuntimeError('Expected source rejection not reproduced')
            else:
                desired=None;indices=None
                with np.load(ALIGN_PARENT/'align_capped/alignment_commands.npz',allow_pickle=False) as z:
                    dense=z['dense'].copy();indices=z['indices'].copy();desired=z['desired'].copy()
                np.savez_compressed(folder/'alignment_commands.npz',dense=dense,indices=indices,desired=desired)
                for i in range(1):
                    if branch.startswith('align') and desired is None:break
                    target=base[i] if desired is None else desired[min(i,len(desired)-1)]
                    action=base[i]+np.clip(target-base[i],-CAPS,CAPS) if branch=='align_capped' else target
                    before=signature(obs);execute(action);executed+=1;check=inspect()
                    drift=float(np.linalg.norm(np.asarray(check['measured_native_ee_pose'][:3])-anchor))
                    control_matches=None
                    if branch=='base_tail':
                        ref=trace[52+i];control_matches=signature(obs)==ref['after_observation'] and state_snapshot(task)==ref['offline_only']['physical']
                        if not control_matches:raise RuntimeError('Base tail replay diverged')
                    row=dict(step=i,slot=20+i,native_count=int(task.take_action_cnt),base=base[i].tolist(),desired=target.tolist(),action=action.tolist(),before_observation=before,after_observation=signature(obs),inspection=check,anchor_drift_m=drift,ready=bool(check['compatible']),control_matches=control_matches,diagnostics=diagnostics(task),physics_steps=guard.steps)
                    with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    if signature(obs)!=alignment_row['after_observation'] or state_snapshot(task)!=alignment_row['diagnostics']['physical'] or row['action']!=alignment_row['action']:
                        raise RuntimeError('First capped state replay differs from parent')
                    if branch.startswith('align') and row['ready']:
                        value=plan(goal,constraint,'resumed_approach');accepted=value['status']=='Success';reason='measured_ready';break
                    if task.eval_success:reason='native_success';break
            record=dict(branch=branch,executed=executed,native_count=int(task.take_action_cnt),reason=reason,original_approach_accepted=accepted,final_inspection=inspect(),native_success=bool(task.eval_success),censored_episode=True,physics_steps=guard.steps)
            write_json(folder/'result.json',record);reports.append(record);print('BRANCH_COMPLETE',branch,json.dumps(record),flush=True)
            task.scene=guard.scene;task.close_env(clear_cache=False);task=guard=None
        write_json(run/'result.json',dict(status='complete',reports=reports,scope=job['scope']))
        replace_json(run/'progress.json',dict(phase='complete',reports=reports));print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),reports=reports));raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task,'scene'):
            if guard is not None:task.scene=guard.scene
            task.close_env(clear_cache=False)

if __name__=='__main__':main()
