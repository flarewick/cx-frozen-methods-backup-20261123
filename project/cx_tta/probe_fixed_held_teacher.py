"""S2d: remaining fixed native switch recovery from newly audited independent source scenes."""
import argparse
import importlib
import json
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from .audit import digest
from .collect_recovery import signature, save_observation, rng_json
from .recovery_contract import rng_capture
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all
from .native_teacher import NativeTeacher, TeacherStopped
from .paths import ROOT, new_run, write_json, replace_json
from .replay_active_objects import compare_source_physical
from .held_teacher import enable_held_continuation
from .probe_action_replay import load_source, source_runtime, source_scene_index, prime_warp_cache, compare_physical


def diagnostics(task):
    """Privileged audit only, never a student input."""
    contacts = sorted({tuple(sorted(c.bodies[i].entity.name for i in (0, 1)))
                       for c in task.scene.get_contacts() if len(c.points)})
    answer = dict(physical=state_snapshot(task), contacts=contacts,
                  grippers_open=[bool(task.is_left_gripper_open()), bool(task.is_right_gripper_open())])
    if hasattr(task, 'switch'):
        answer['switch_joint_limits'] = np.asarray(task.switch.get_qlimits()).tolist()
    return answer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--point', type=int, required=True)
    parser.add_argument('--repeat',type=int,choices=(0,1),default=0)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if not args.gpu.isdigit(): parser.error('GPU index required')
    protocol = json.loads(args.protocol.read_text())
    if str(args.point) not in protocol['held_sources'] or args.repeat != 0:
        raise ValueError('Only remaining fixed switch points, one attempt each')
    point = protocol['points'][args.point]
    if point['task'] != 'place_shoe':
        raise ValueError('Task teacher entry has not been qualified')
    for path, expected in {**protocol['source_sha256'], **protocol['code_sha256']}.items():
        if digest(Path(path)) != expected: raise ValueError('Frozen dependency changed: '+path)
    support_run=Path(protocol['held_sources'][str(args.point)])
    support=json.loads((support_run/'result.json').read_text());sa=json.loads((support_run/'audit.json').read_text())
    if not sa['passed'] or not support['held_qualified']:raise ValueError('Audited held entry required')
    owner=support['owner'];support_fork=json.loads((support_run/'hold/fork.json').read_text())
    source = Path(point['source'])
    manifest, episode, rows = load_source(source, point['seed'])
    runtime = source_runtime(source)
    config = json.loads((source/'resolved_config.json').read_text())
    stride = config['save_freq']
    if not isinstance(stride, int) or stride < 1: raise ValueError('Invalid frozen planner stride')
    run = new_run('fixed_held', group=('recovery_teachers', point['task']))
    print(run, flush=True)
    write_json(run/'manifest.json', dict(point=args.point, repeat=args.repeat, teacher_seed=protocol['teacher_seed'], protocol=str(args.protocol.resolve()),
        protocol_sha256=digest(args.protocol), source=str(source), scene_seed=point['seed'],
        instruction=point['instruction'], prefix_actions=point['prefix_actions'],
        teacher='official play_once with verified held entry: retain closure, skip pregrasp/approach; native lift, preplace, place and release',support_run=str(support_run),verified_owner=owner,
        dense_stride=stride, dense_stride_source='frozen source config save_freq; NOT physical time equivalence',
        selection='Every support-qualified predeclared shoe fork, one fresh-process attempt each; no best-of-N',
        privileged=True, training_dataset=False, video=False,
        observation_saving='fork, every 32 native actions, each semantic plan end, terminal; no per-action image dataset',
        code_sha256={str(p): digest(p) for p in (Path(__file__), ROOT/'cx_tta/native_teacher.py', ROOT/'cx_tta/held_teacher.py', ROOT/'cx_tta/replay_active_objects.py',
            runtime/'envs/_base_task.py', runtime/'envs'/f"{point['task']}.py")}))
    task = teacher = None
    started = time.monotonic()
    def timed_out(*_): raise TimeoutError('Teacher probe wall budget reached (900 seconds)')
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(900)
    try:
        prime_warp_cache(source, run)
        configure_runtime(run, runtime, resolve_tokenizer())
        task = getattr(importlib.import_module('envs.'+point['task']), point['task'])()
        seed_all(point['seed'])
        config.update(save_path=str(run/'runtime_data'), cx_runtime=str(runtime),
                      save_data=False, eval_video_log=False, eval_video_save_dir=None)
        task.setup_demo(now_ep_num=source_scene_index(manifest, point['seed']), seed=point['seed'],
                        **dict(config, need_plan=True, is_test=True))
        task.set_instruction(instruction=point['instruction'])
        obs = task.get_obs()
        for chunk in range(point['chunk']):
            if signature(obs) != rows[chunk]['observation_components']:
                raise RuntimeError('Prefix RGB/command mismatch')
            for action in rows[chunk]['base_executed']:
                count = task.take_action_cnt
                task.take_action(np.asarray(action, dtype=np.float32))
                obs = task.get_obs()
                if task.take_action_cnt != count+1: raise RuntimeError('Prefix stopped early')
        physical = compare_source_physical(source, point['seed'], rows[point['chunk']]['before'], task, state_snapshot(task), 1e-5)
        if not physical['matched'] or signature(obs) != point['observation_signature']:
            write_json(run/'replay_mismatch.json',dict(physical=physical,actual=signature(obs),expected=point['observation_signature']))
            raise RuntimeError('Teacher fork differs from original')
        if task.take_action_cnt != point['prefix_actions'] or task.eval_success:
            raise RuntimeError('Wrong budget or terminal fork')
        if episode['success'] or not episode['eligible'] or task.check_success():
            raise ValueError('Require an eligible failed L0 scene and currently unsuccessful fork')
        prefix, native_limit = task.take_action_cnt, task.step_lim
        save_observation(run/'fork.npz', obs)
        write_json(run/'fork.json', dict(physical_match=physical, observation=signature(obs),
            native_limit=native_limit, remaining=native_limit-prefix, diagnostics=diagnostics(task)))
        previous_signature = signature(obs)
        actions = [];stages=[]
        def on_plan(index, dense, indices, control):
            expected=('closure','lift','preplace','place','release','release')[index] if index<6 else 'unsupported'
            other='right' if owner=='left' else 'left'
            if control[other+'_arm'] is not None or control[other+'_gripper'] is not None:raise ValueError('Unverified owner control')
            arm=control[owner+'_arm'];grip=control[owner+'_gripper']
            if expected in ('lift','preplace','place'):
                if arm is None or grip is not None:raise ValueError('Held teacher stage mismatch')
            elif expected in ('closure','release'):
                target=0 if expected=='closure' else 1
                if arm is not None or grip is None or abs(float(grip['result'][-1])-target)>1e-6:raise ValueError('Held gripper stage mismatch')
            else:raise ValueError('Unexpected teacher semantic plan')
            stages.append(expected)
            write_json(run/f'semantic_plan_{index:03d}.json',dict(stage=expected,owner=owner,native_count=int(task.take_action_cnt)))
            np.savez_compressed(run/f'plan_{index:03d}.npz', dense_commands=dense, indices=indices)
            replace_json(run/'progress.json', dict(phase='teacher', plan=index,
                action_count=task.take_action_cnt, native_limit=native_limit, seconds=time.monotonic()-started))
        def on_action(plan_id, dense_index, action, observation, physics_steps, plan_end):
            nonlocal previous_signature
            actions.append(action.copy())
            next_signature = signature(observation)
            row = dict(plan=plan_id, semantic_stage=stages[plan_id], dense_index=dense_index, action=action.tolist(),
                action_count=task.take_action_cnt, physics_steps=physics_steps,
                before_observation=previous_signature, after_observation=next_signature,
                success=bool(task.eval_success), offline_only=diagnostics(task))
            with (run/'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            previous_signature = next_signature
            if len(actions)%32 == 0 or plan_end or task.eval_success:
                save_observation(run/f'observation_{len(actions):04d}.npz', observation)
        write_json(run/'history_context.json', dict(past_observation=point['past_observation'],
            past_observation_sha256=point['past_observation_sha256'],
            subsequent_history='Strictly previous teacher observation at 32 actual teacher actions; no future L0 inputs'))
        if signature(obs)!=support_fork['observation'] or state_snapshot(task)!=support_fork['diagnostics']['physical']:
            raise ValueError('Held support fork changed')
        if owner!=support_fork['teacher_selected_arm']:raise ValueError('Original teacher selects a different arm than verified owner')
        enable_held_continuation(task,task.shoe.get_name(),owner,lambda entry:write_json(run/'stage_entry.json',entry))
        seed_all(protocol['teacher_seed'])
        write_json(run/'teacher_rng.json',rng_json(rng_capture()))
        (run/'trace.jsonl').touch(exist_ok=False)
        teacher = NativeTeacher(task, stride, on_action, on_plan)
        try:
            task.play_once()
            reason = 'teacher_script_finished' if task.plan_success else 'teacher_planning_failed'
        except TeacherStopped as exc:
            reason = str(exc)
        save_observation(run/'terminal.npz', task.get_obs())
        np.savez_compressed(run/'native_actions.npz', actions=np.asarray(actions, dtype=np.float32).reshape(-1,14))
        success = bool(task.eval_success)
        terminal = success or task.take_action_cnt == native_limit
        if teacher.executed != len(actions) or task.take_action_cnt != prefix+len(actions):
            raise RuntimeError('Teacher accounting mismatch')
        result = dict(status='complete_probe', reason=reason, teacher_success=success,
            native_terminal=terminal, terminal_success=success if terminal else None,
            plan_success=bool(task.plan_success), teacher_plans=teacher.plans,
            action_count=task.take_action_cnt, native_limit=native_limit, prefix_actions=prefix,
            teacher_actions=len(actions), observation_updates=len(actions), physics_steps=teacher.guard.steps,
            physics_dt=float(task.scene.get_timestep()), fork_verified=True,
            final_diagnostics=diagnostics(task), seconds=time.monotonic()-started,
            scope='Privileged teacher feasibility at one developer state; not adapter improvement or bounded-residual feasibility.')
        if (run/'trace.jsonl').exists(): result['trace_sha256'] = digest(run/'trace.jsonl')
        write_json(run/'result.json', result)
        from .audit_teacher_probe import audit
        checked = audit(run)
        obj=np.asarray(state_snapshot(task)['offline_only']['object_poses']['shoe']);quat=obj[3:].copy()
        if quat[0]<0:quat*=-1
        recomputed=bool(np.all(np.abs(obj[:2]-[0,-.08])<[.05,.02]) and np.all(np.abs(quat-[.5,.5,-.5,-.5])<.07) and task.is_left_gripper_open() and task.is_right_gripper_open())
        semantic_completed=len(stages)>=5 and stages[-1]=='release'
        qualified=bool(checked['passed'] and success and terminal and task.plan_success and reason=='native_success'
                       and recomputed and task.check_success() and semantic_completed)
        write_json(run/'qualification.json',dict(qualified=qualified,interface_passed=checked['passed'],
            original_predicate_passed=bool(task.check_success()),official_success_recomputed=recomputed,
            verified_owner=owner,support_qualified=True,semantic_completed=semantic_completed,stages=stages,
            teacher_actions=len(actions),training_cache_ready=False,
            scope='Verified held-entry native teacher only; not original-cap or adapter benefit'))
        replace_json(run/'progress.json', dict(phase='complete', reason=reason, teacher_success=success,qualified=qualified))
        if not checked['passed']: raise RuntimeError('Teacher accounting audit failed')
        print(json.dumps({k:v for k,v in result.items() if k!='final_diagnostics'}), flush=True)
    except BaseException:
        write_json(run/'failure.json', dict(traceback=traceback.format_exc(), seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0)
        if task is not None and hasattr(task, 'scene'):
            if teacher is not None: task.scene = teacher.guard.scene
            task.close_env()


if __name__ == '__main__':
    main()
