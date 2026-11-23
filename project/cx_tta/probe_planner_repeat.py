"""Fixed-state/fixed-goal native planning repeats without VLA or rollout scoring."""
import argparse
import importlib
import json
import signal
import traceback
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_place_goal import BOUNDED
from .collect_recovery import DEFAULT_PROTOCOL, signature
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
from .native_teacher import SceneStepGuard, export_commands
from .paths import ROOT,new_run,write_json,replace_json
from .probe_action_replay import load_source, source_runtime, source_scene_index, prime_warp_cache
from .recovery_contract import rng_capture,rng_restore,rng_fingerprint
from .teacher_projection import planner_status,diagnostic_value

def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not a.gpu.isdigit():p.error('GPU index required')
    if not read(BOUNDED/'audit.json')['passed']:raise ValueError('Complete source action audit required')
    point=read(DEFAULT_PROTOCOL)['points'][4];source=Path(point['source']);manifest,episode,rows=load_source(source,point['seed']);runtime=source_runtime(source)
    config=read(source/'resolved_config.json')
    conditions=[('restart_64',BOUNDED/'compatible/plan_0000.json',[]),('held_preplacement_94',BOUNDED/'held_compatible/plan_0002.json',[json.loads(x) for x in (BOUNDED/'held_compatible/trace.jsonl').read_text().splitlines() if json.loads(x)['advanced']][:30])]
    files=[DEFAULT_PROTOCOL,BOUNDED/'audit.json',BOUNDED/'held_compatible/trace.jsonl',*(c[1] for c in conditions),source/'resolved_config.json',Path(__file__),runtime/'envs/robot/planner.py',runtime/'envs/robot/robot.py']
    job=dict(source=str(source),scene_seed=point['seed'],native_budget=500,checkpoints=[64,94],fixed_global_planning_seed=920004,
        repeats='3 sequential calls on each of 3 fresh identical scenes per checkpoint: 18 plans. Restore exact Python/NumPy/Torch CPU/CUDA RNG before every call; native planner internal state NOT reset within scene.',
        scope='Native planning-only developer probe; no VLA query, training collection, simulation after planning, or task success score',video=False,source_sha256={str(f):digest(f) for f in files})
    if a.dry_run:print(json.dumps(job,indent=2));return
    run=new_run('probe',group=('planner_repeatability','place_shoe'));print(run,flush=True);write_json(run/'manifest.json',job)
    task=guard=None;records=[]
    def timeout(*args):raise TimeoutError('Planning-repeat probe exceeded 1500 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1500)
    try:
        prime_warp_cache(source,run);configure_runtime(run,runtime,resolve_tokenizer())
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.set_num_threads(1);cls=getattr(importlib.import_module('envs.place_shoe'),'place_shoe')
        config.update(save_path=str(run/'runtime_data'),cx_runtime=str(runtime),save_data=False,eval_video_log=False,eval_video_save_dir=None)
        for name,planpath,tail in conditions:
            expected=read(planpath);pose=expected['goal']['pose'];constraint=expected['goal']['constraint']
            for scene in range(3):
                folder=run/name/f'scene_{scene}';folder.mkdir(parents=True)
                replace_json(run/'progress.json',dict(checkpoint=name,scene=scene,phase='replaying',completed=len(records)))
                task=cls();seed_all(point['seed']);task.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']),seed=point['seed'],**dict(config,need_plan=True,is_test=True));task.set_instruction(instruction=point['instruction']);obs=task.get_obs()
                guard=SceneStepGuard(task.scene);task.scene=guard
                def execute(action):
                    nonlocal obs
                    before=task.take_action_cnt;guard.in_native=True
                    try:task.take_action(np.asarray(action,np.float32))
                    finally:guard.in_native=False
                    obs=task.get_obs()
                    if task.take_action_cnt!=before+1 or task.step_lim!=500:raise RuntimeError('Replay budget changed')
                for c in range(point['chunk']):
                    if signature(obs)!=rows[c]['observation_components']:raise RuntimeError('L0 prefix mismatch')
                    for action in rows[c]['base_executed']:execute(action)
                if signature(obs)!=point['observation_signature']:raise RuntimeError('Source fork mismatch')
                for row in tail:
                    if signature(obs)!=row['before_observation']:raise RuntimeError('Held replay predecessor mismatch')
                    execute(row['action'])
                    if signature(obs)!=row['after_observation'] or state_snapshot(task)!=row['offline_only']['physical']:raise RuntimeError('Held replay divergence')
                if task.take_action_cnt!=expected['native_count'] or state_snapshot(task)!=expected['physical'] or signature(obs)!=expected['observation']:raise RuntimeError('Fixed planning checkpoint mismatch')
                if not (run/'actor_metadata.json').exists():write_json(run/'actor_metadata.json',dict(scene_seed=point['seed'],model_id=int(task.shoe_id),config=diagnostic_value(task.shoe.config)))
                write_json(folder/'fork.json',dict(physical=state_snapshot(task),observation=signature(obs),native_count=int(task.take_action_cnt),physics_steps=guard.steps,measured_ee_pose=diagnostic_value(task.robot.get_right_ee_pose()),target_functional_pose=diagnostic_value(task.target_block.get_functional_point(0))))
                seed_all(920004);fixed=rng_capture()
                for repeat in range(3):
                    rng_restore(fixed);before=(state_snapshot(task),signature(task.get_obs()),int(task.take_action_cnt),guard.steps,task.plan_success)
                    pre_rng=rng_fingerprint(rng_capture())
                    with planner_status(task,'right') as status:value=task.robot.right_plan_path(pose,constraint_pose=constraint)
                    post_rng=rng_fingerprint(rng_capture());after=(state_snapshot(task),signature(task.get_obs()),int(task.take_action_cnt),guard.steps,task.plan_success)
                    if after!=before:raise RuntimeError('Planner mutated simulation, budget, observations or task flag')
                    record=dict(checkpoint=name,scene=scene,repeat=repeat,goal=pose,constraint=constraint,status=value['status'],motion_gen=status,rng_before=pre_rng,rng_after=post_rng,physical_unchanged=True,native_count=int(task.take_action_cnt))
                    if value['status']=='Success':
                        control={k:None for k in ('left_arm','right_arm','left_gripper','right_gripper')};control['right_arm']=value
                        dense,indices=export_commands(control,task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),config['save_freq'])
                        file=folder/f'plan_{repeat}.npz';np.savez_compressed(file,position=value['position'],velocity=value['velocity'],dense=dense,indices=indices,commands=dense[indices]);record.update(file=str(file),sha256=digest(file),dense_length=len(dense),native_command_length=len(indices))
                    write_json(folder/f'plan_{repeat}.json',record);records.append(record)
                    print('PLAN_COMPLETE',name,scene,repeat,value['status'],record.get('dense_length'),flush=True)
                task.scene=guard.scene;task.close_env(clear_cache=False);task=guard=None
        write_json(run/'result.json',dict(status='complete',records=records,scope=job['scope']));replace_json(run/'progress.json',dict(phase='complete',plans=len(records)));print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),records=records));raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task,'scene'):
            if guard is not None:task.scene=guard.scene
            task.close_env(clear_cache=False)

if __name__=='__main__':main()
