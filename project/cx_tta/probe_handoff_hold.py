"""S1i: bounded holding/lifting checks at an exact policy handoff state.

Privileged diagnostic; no new VLA query, video or training data. A frozen base
block is legal only within its first 32 callbacks, hence the 12-action window.
"""
import argparse,importlib,json,signal,traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix,error
from .collect_recovery import DEFAULT_PROTOCOL,signature
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import configure_runtime,resolve_tokenizer,seed_all
from .native_teacher import SceneStepGuard,export_commands
from .paths import ROOT,new_run,write_json,replace_json
from .probe_action_replay import load_source,source_runtime,source_scene_index,prime_warp_cache
from .probe_recovery_teacher import diagnostics
from .robot_kinematics import RobotFK
from .teacher_projection import CAPS,planner_status
from .handoff_support import matching_links

PARENT=ROOT/'outputs/compatible_continuations/place_shoe/gate_20261001_141842_178427'
BRANCHES=('hold_native','lift_native','lift_capped')
WINDOW=12
LIFT_M=.01
RELATIVE_POSITION_M=.003
RELATIVE_ANGLE_DEG=3.

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not args.gpu.isdigit():p.error('GPU index required')
    parent=json.loads((PARENT/'manifest.json').read_text())
    if not json.loads((PARENT/'audit.json').read_text())['passed']:raise ValueError('Parent audit failed')
    protocol=json.loads(DEFAULT_PROTOCOL.read_text());point=protocol['points'][4];source=Path(point['source'])
    manifest,episode,rows=load_source(source,point['seed']);runtime=source_runtime(source)
    for file,sha in [(source/'manifest.json',point['source_manifest_sha256']),(source/'resolved_config.json',point['source_config_sha256']),(source/'traces'/f"r0_seed{point['seed']}.jsonl",point['source_trace_sha256'])]:
        if digest(file)!=sha:raise ValueError('Source changed')
    config=json.loads((source/'resolved_config.json').read_text());fk=RobotFK(config,'right')
    with np.load(PARENT/'common_base.npz',allow_pickle=False) as z:base=z['base'].copy()
    expected=json.loads((PARENT/'base/fork.json').read_text())
    job=dict(task=point['task'],scene_seed=point['seed'],source=str(source),parent=str(PARENT),branches=BRANCHES,prefix_actions=64,native_limit=500,window=WINDOW,lift_m=LIFT_M,caps=CAPS.tolist(),planner_stride=config['save_freq'],video=False,
        relative_thresholds=dict(position_m=RELATIVE_POSITION_M,orientation_deg=RELATIVE_ANGLE_DEG),lift_thresholds=dict(measured_ee_rise_m=.008,object_rise_m=.005,goal_error_m=.005,orientation_deg=3.),
        scope='local support/motion test, not force closure, complete recovery, adapter evaluation or grasp dataset',
        gripper='preserve current normalized command; actual finger qpos logged separately',
        source_sha256={str(f):digest(f) for f in [PARENT/'audit.json',PARENT/'manifest.json',PARENT/'common_base.npz',PARENT/'base/fork.json',DEFAULT_PROTOCOL,source/'resolved_config.json',fk.urdf]})
    if args.dry_run:print(json.dumps(job,indent=2));return
    run=new_run('probe',group=('handoff_support',point['task']));print(run,flush=True)
    files=[Path(__file__),ROOT/'cx_tta/handoff_support.py',ROOT/'cx_tta/native_teacher.py',ROOT/'cx_tta/robot_kinematics.py',ROOT/'cx_tta/audit_local_geometry.py',runtime/'envs/robot/robot.py',runtime/'envs/robot/planner.py',runtime/'envs/_base_task.py']
    write_json(run/'manifest.json',dict(**job,code_sha256={str(f):digest(f) for f in files}))
    task=guard=None;reports=[]
    def timeout(*_):raise TimeoutError('Local holding probe exceeded1200seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    try:
        prime_warp_cache(source,run);configure_runtime(run,runtime,resolve_tokenizer());import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.set_num_threads(1);cls=getattr(importlib.import_module('envs.'+point['task']),point['task'])
        config.update(save_path=str(run/'runtime_data'),cx_runtime=str(runtime),save_data=False,eval_video_log=False,eval_video_save_dir=None)
        for branch in BRANCHES:
            folder=run/branch;folder.mkdir();replace_json(run/'progress.json',dict(branch=branch,phase='replaying',reports=reports))
            task=cls();seed_all(point['seed']);task.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']),seed=point['seed'],**dict(config,need_plan=True,is_test=True));task.set_instruction(instruction=point['instruction']);obs=task.get_obs()
            guard=SceneStepGuard(task.scene);task.scene=guard
            def execute(command):
                nonlocal obs
                n=task.take_action_cnt
                if task.step_lim!=500 or n>=76 or task.eval_success:raise RuntimeError('Probe budget/terminal changed')
                guard.in_native=True
                try:task.take_action(np.asarray(command,np.float32))
                finally:guard.in_native=False
                obs=task.get_obs()
                if task.take_action_cnt!=n+1:raise RuntimeError('Native action did not advance')
            for c in range(point['chunk']):
                if signature(obs)!=rows[c]['observation_components']:raise RuntimeError('Prefix observation mismatch')
                for command in rows[c]['base_executed']:execute(command)
            if task.take_action_cnt!=64 or signature(obs)!=expected['observation'] or state_snapshot(task)!=expected['diagnostics']['physical']:raise RuntimeError('Exact fork mismatch')
            # Derive finger links from robot joints, not task-specific link rules.
            elements={j.get('name'):j.find('child').get('link') for j in ET.parse(fk.urdf).getroot().findall('joint')}
            finger_links=[elements[j[0].get_name()] for j in task.robot.right_gripper if j[0] is not None]
            object_name=task.shoe.get_name()
            initial_ee=np.asarray(task.robot.get_right_ee_pose(),float);initial_object=pose_matrix(state_snapshot(task)['offline_only']['object_poses']['shoe']);initial_relative=np.linalg.inv(pose_matrix(initial_ee))@initial_object
            current=np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),np.float32)
            goal=initial_ee.copy();goal[2]+=LIFT_M
            write_json(folder/'fork.json',dict(native_count=64,observation=signature(obs),diagnostics=diagnostics(task),measured_ee_pose=initial_ee.tolist(),object_matrix=initial_object.tolist(),relative_matrix=initial_relative.tolist(),current_command=current.tolist(),finger_links=finger_links,object_name=object_name))
            desired=np.tile(current,(WINDOW,1));indices=None
            if branch!='hold_native':
                before=(state_snapshot(task),guard.steps,task.take_action_cnt,task.plan_success)
                with planner_status(task,'right') as detail:plan=task.robot.right_plan_path(goal.tolist(),constraint_pose=[1,1,1,0,0,0])
                if before!=(state_snapshot(task),guard.steps,task.take_action_cnt,task.plan_success):raise RuntimeError('Planning mutated physical state or budget')
                write_json(folder/'planning.json',dict(goal=goal.tolist(),constraint=[1,1,1,0,0,0],status=plan['status'],motion_gen=detail))
                if plan['status']!='Success':raise RuntimeError('Small lift planning failed; experiment incomplete')
                control=dict(left_arm=None,right_arm=plan,left_gripper=None,right_gripper=None)
                dense,indices=export_commands(control,current,config['save_freq']);commands=dense[indices]
                desired=np.stack([commands[min(i,len(commands)-1)] for i in range(WINDOW)])
                np.savez_compressed(folder/'plan.npz',dense=dense,indices=indices,desired=desired)
            recorded=[]
            for i in range(WINDOW):
                action=base[i]+np.clip(desired[i]-base[i],-CAPS,CAPS) if branch=='lift_capped' else desired[i]
                before_obs=signature(obs);ticks=guard.steps;execute(action);d=diagnostics(task);ee=pose_matrix(task.robot.get_right_ee_pose());obj=pose_matrix(d['physical']['offline_only']['object_poses']['shoe']);relative=np.linalg.inv(ee)@obj
                slip=error(relative,initial_relative);target=pose_matrix(initial_ee if branch=='hold_native' else goal)
                contacts=matching_links(d['contacts'],object_name,finger_links)
                row=dict(step=i,native_count=int(task.take_action_cnt),action=action.tolist(),desired=desired[i].tolist(),base=base[i].tolist(),before_observation=before_obs,after_observation=signature(obs),physics_steps=guard.steps-ticks,measured_ee_matrix=ee.tolist(),object_matrix=obj.tolist(),relative_matrix=relative.tolist(),relative_error=slip,target_error=error(ee,target),ee_rise_m=float(ee[2,3]-initial_ee[2]),object_rise_m=float(obj[2,3]-initial_object[2,3]),finger_contacts=contacts,diagnostics=d)
                with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                recorded.append(row)
            final=recorded[-1];coupled=all(x['relative_error']['distance_m']<=RELATIVE_POSITION_M and x['relative_error']['angle_deg']<=RELATIVE_ANGLE_DEG and len(x['finger_contacts'])==len(finger_links) for x in recorded)
            arrival=final['target_error']['distance_m']<=.005 and final['target_error']['angle_deg']<=3.
            passed=coupled and arrival and (branch=='hold_native' or final['ee_rise_m']>=.008 and final['object_rise_m']>=.005)
            record=dict(branch=branch,executed=WINDOW,native_count=76,local_support_passed=bool(passed),coupling_passed=bool(coupled),arrival_passed=bool(arrival),final_ee_rise_m=final['ee_rise_m'],final_object_rise_m=final['object_rise_m'],max_relative_drift_m=max(x['relative_error']['distance_m'] for x in recorded),max_relative_angle_deg=max(x['relative_error']['angle_deg'] for x in recorded),both_finger_contact_actions=sum(len(x['finger_contacts'])==len(finger_links) for x in recorded),physics_steps=guard.steps,censored_episode=True,native_success=bool(task.eval_success),trace_sha256=digest(folder/'trace.jsonl'))
            write_json(folder/'result.json',record);reports.append(record);print('BRANCH_COMPLETE',json.dumps(record),flush=True)
            task.scene=guard.scene;task.close_env(clear_cache=False);task=guard=None
        write_json(run/'result.json',dict(status='complete',reports=reports,scope=job['scope']));replace_json(run/'progress.json',dict(phase='complete',reports=reports));print('COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),reports=reports));raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task,'scene'):
            if guard is not None:task.scene=guard.scene
            task.close_env(clear_cache=False)
if __name__=='__main__':main()
