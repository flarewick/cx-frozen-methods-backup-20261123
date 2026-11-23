"""S2d privileged support assay at four frozen forks; no training labels."""
import argparse
import importlib
import json
import signal
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix, error
from .collect_recovery import signature
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
from .native_teacher import SceneStepGuard, export_commands
from .paths import ROOT, new_run, write_json
from .probe_action_replay import load_source, source_runtime, source_scene_index, prime_warp_cache, compare_physical
from .probe_multiscene_teacher import diagnostics
from .robot_kinematics import RobotFK
from .replay_active_objects import compare_source_physical


def finger_entities(task, config, side):
    fk = RobotFK(config, side)
    names = {j.get('name'): j.find('child').get('link') for j in ET.parse(fk.urdf).getroot().findall('joint')}
    articulation = getattr(task.robot, side+'_entity')
    return [articulation.find_link_by_name(names[j[0].get_name()]).entity
            for j in getattr(task.robot, side+'_gripper') if j[0] is not None]


def contact_evidence(task, fingers):
    # Match actual entities, not names: the two robot instances can share link names.
    contacts = [c for c in task.scene.get_contacts() if len(c.points)]
    result = {}
    object_name = task.shoe.get_name()
    for side, entities in fingers.items():
        result[side] = [any((c.bodies[0].entity == entity and c.bodies[1].entity.name == object_name)
                           or (c.bodies[1].entity == entity and c.bodies[0].entity.name == object_name)
                           for c in contacts) for entity in entities]
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--point', type=int, required=True)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    if args.point not in p['shoe_support_points'] or not args.gpu.isdigit(): raise ValueError('Unfrozen point/GPU')
    for f,h in {**p['source_sha256'], **p['code_sha256']}.items():
        if digest(Path(f)) != h: raise ValueError('Frozen dependency changed '+f)
    point = p['points'][args.point]; source = Path(point['source'])
    manifest, episode, rows = load_source(source, point['seed'])
    runtime = source_runtime(source); config = json.loads((source/'resolved_config.json').read_text())
    run = new_run('fixed', group=('handoff_support', 'place_shoe')); print(run, flush=True)
    write_json(run/'manifest.json', dict(protocol=str(args.protocol.resolve()), protocol_sha256=digest(args.protocol),
        point=args.point, source=str(source), scene_seed=point['seed'], prefix_actions=point['prefix_actions'],
        branches=['hold', 'lift'], video=False, teacher_labels=False, code_sha256=p['code_sha256']))
    task = guard = None; reports = []
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Shoe assay 900s budget')))
    signal.alarm(900)
    try:
        prime_warp_cache(source, run); configure_runtime(run, runtime, resolve_tokenizer())
        cls = getattr(importlib.import_module('envs.place_shoe'), 'place_shoe')
        config.update(save_path=str(run/'runtime_data'), cx_runtime=str(runtime), save_data=False,
                      eval_video_log=False, eval_video_save_dir=None)
        for branch in ('hold', 'lift'):
            task = cls(); seed_all(point['seed'])
            task.setup_demo(now_ep_num=source_scene_index(manifest,point['seed']), seed=point['seed'],
                            **dict(config, need_plan=True, is_test=True))
            task.set_instruction(instruction=point['instruction']); obs = task.get_obs()
            for i in range(point['chunk']):
                if signature(obs) != rows[i]['observation_components']:
                    write_json(run/'replay_mismatch.json',dict(stage='prefix',chunk=i,actual=signature(obs),expected=rows[i]['observation_components'],physical=compare_physical(rows[i]['before'],state_snapshot(task),1e-5)))
                    raise ValueError('Prefix observation mismatch')
                for action in rows[i]['base_executed']:
                    count = task.take_action_cnt; task.take_action(np.asarray(action,np.float32)); obs = task.get_obs()
                    if task.take_action_cnt != count+1: raise ValueError('Stopped prefix')
            match = compare_source_physical(source, point['seed'], rows[point['chunk']]['before'], task, state_snapshot(task), 1e-5)
            write_json(run/f'{branch}_replay_check.json',dict(stage='fork',physical=match,actual=signature(obs),expected=point['observation_signature'],actual_physical=state_snapshot(task),expected_physical=rows[point['chunk']]['before']))
            if not match['matched'] or signature(obs) != point['observation_signature']:
                write_json(run/'replay_mismatch.json',dict(stage='fork',physical=match,observation_exact=signature(obs)==point['observation_signature']))
                raise ValueError('Fork mismatch')
            if task.take_action_cnt != point['prefix_actions'] or task.eval_success or task.check_success(): raise ValueError('Invalid fork')
            fingers = {s: finger_entities(task,config,s) for s in ('left','right')}
            contacts = contact_evidence(task, fingers)
            owners = [s for s in fingers if len(fingers[s])==2 and all(contacts[s])]
            fork = dict(physical_match=match, observation=signature(obs), diagnostics=diagnostics(task),
                        finger_contacts=contacts, finger_entity_ids={s:[int(e.per_scene_id) for e in es] for s,es in fingers.items()},
                        candidate_owners=owners, ee_pose_by_side={s:np.asarray(getattr(task.robot,'get_'+s+'_ee_pose')(),float).tolist() for s in ('left','right')}, teacher_selected_arm='left' if task.shoe.get_pose().p[0]<0 else 'right')
            folder=run/branch;folder.mkdir();write_json(folder/'fork.json',fork)
            if len(owners)!=1:
                reports.append(dict(branch=branch, passed=False, reason='no_unique_two_finger_owner', fork_verified=True))
                task.close_env();task=None
                break
            side=owners[0]; initial_ee=np.asarray(getattr(task.robot,'get_'+side+'_ee_pose')(),float)
            obj0=pose_matrix(state_snapshot(task)['offline_only']['object_poses']['shoe']); rel0=np.linalg.inv(pose_matrix(initial_ee))@obj0
            current=np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),np.float32)
            goal=initial_ee.copy();goal[2]+=(.01 if branch=='lift' else 0)
            guard=SceneStepGuard(task.scene); task.scene=guard
            desired=np.tile(current,(12,1))
            seed_all(p['teacher_seed'])
            if branch=='lift':
                before=state_snapshot(task); plan=getattr(task.robot,side+'_plan_path')(goal.tolist(),constraint_pose=[1,1,1,0,0,0])
                if before != state_snapshot(task) or guard.steps: raise ValueError('Planning changed physical state')
                write_json(folder/'planning.json',dict(status=plan['status'],goal=goal.tolist()))
                if plan['status']!='Success':
                    reports.append(dict(branch=branch,passed=False,reason='lift_planning_failed',owner=side,fork_verified=True));break
                control=dict(left_arm=None,right_arm=None,left_gripper=None,right_gripper=None);control[side+'_arm']=plan
                dense,indices=export_commands(control,current,config['save_freq']);commands=dense[indices]
                desired=np.stack([commands[min(i,len(commands)-1)] for i in range(12)])
                np.savez_compressed(folder/'plan.npz',dense=dense,indices=indices,desired=desired)
            recorded=[]
            for i,action in enumerate(desired):
                count=task.take_action_cnt
                if count>=task.step_lim or task.eval_success: break
                previous=signature(obs); ticks=guard.steps;guard.in_native=True
                try:task.take_action(action)
                finally:guard.in_native=False
                obs=task.get_obs()
                if task.take_action_cnt!=count+1:raise ValueError('Native accounting failed')
                ee=pose_matrix(getattr(task.robot,'get_'+side+'_ee_pose')());obj=pose_matrix(state_snapshot(task)['offline_only']['object_poses']['shoe'])
                row=dict(step=i,action=action.tolist(),native_count=int(task.take_action_cnt),before_observation=previous,
                    after_observation=signature(obs),physics_steps=guard.steps-ticks,finger_contacts=contact_evidence(task,fingers),
                    ee=ee.tolist(),object=obj.tolist(),relative_error=error(np.linalg.inv(ee)@obj,rel0),
                    target_error=error(ee,pose_matrix(goal)),ee_rise=float(ee[2,3]-initial_ee[2]),object_rise=float(obj[2,3]-obj0[2,3]))
                recorded.append(row)
                with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            last=recorded[-1] if recorded else None
            coupled=len(recorded)==12 and all(x['relative_error']['distance_m']<=.003 and x['relative_error']['angle_deg']<=3 and all(x['finger_contacts'][side]) for x in recorded)
            arrival=bool(last and last['target_error']['distance_m']<=.005 and last['target_error']['angle_deg']<=3)
            passed=bool(coupled and arrival and (branch=='hold' or last['ee_rise']>=.008 and last['object_rise']>=.005))
            report=dict(branch=branch,passed=passed,owner=side,fork_verified=True,executed=len(recorded),coupled=bool(coupled),arrival=arrival,
                final_ee_rise=last['ee_rise'] if last else None,final_object_rise=last['object_rise'] if last else None,
                max_drift_m=max((x['relative_error']['distance_m'] for x in recorded),default=None),
                max_drift_deg=max((x['relative_error']['angle_deg'] for x in recorded),default=None),
                original_teacher_arm_matches=side==fork['teacher_selected_arm'])
            write_json(folder/'result.json',report);reports.append(report)
            task.scene=guard.scene;task.close_env();task=guard=None
        qualified=len(reports)==2 and all(x['passed'] for x in reports) and len({x['owner'] for x in reports})==1
        write_json(run/'result.json',dict(status='complete',point=args.point,reports=reports,held_qualified=qualified,
            owner=reports[0].get('owner') if qualified else None,training_cache_ready=False,
            scope='Privileged local support evidence; no full recovery, force-closure or adapter benefit claim'))
        print('SUPPORT_COMPLETE',json.dumps(reports),flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),reports=reports));raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task,'scene'):
            if guard is not None:task.scene=guard.scene
            task.close_env()

if __name__=='__main__':main()
