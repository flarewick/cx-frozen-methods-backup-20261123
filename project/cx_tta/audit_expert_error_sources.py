"""Localize expert imitation errors without changing targets or opening final holdouts."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .audit import digest
from .audit_correction_budget import CAPS
from .dataset import expert_chunk, load_cache
from .effect_contract import timeline
from .paths import new_run, write_json
from .robot_kinematics import RobotFK


def error_components(base, target, valid):
    residual = (target.astype(float) - base) / CAPS
    weight = valid[..., None] / (valid.sum(1)[:, None, None] * 14)
    squared = np.where(valid[..., None], residual**2, 0) * weight
    floor = np.where(valid[..., None], np.maximum(np.abs(residual)-1, 0)**2, 0) * weight
    gradient = np.where(valid[..., None], np.abs(residual), 0) * weight
    return residual, squared, floor, gradient


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    run = new_run('expert_error_sources', group=('reports', 'audits'))
    write_json(run/'protocol.json', dict(plan=str(args.plan.resolve()), plan_sha256=digest(args.plan),
        code_sha256=digest(Path(__file__)), caps=CAPS.tolist(),
        scope='Same 48 fit / 12 inner-held expert episodes as semantic pilot; original validation unopened. Fixed per-channel, horizon 1-4/5-8/9-16/17-32 and episode decomposition; no filtering or cap tuning. Gradient refers to output-space absolute normalized MSE gradient at zero residual, not parameter gradient.'))
    records = []
    for r in plan['records']:
        if r['kind'] != 'expert':
            continue
        if r['split'] != 'train' or r['pilot_split'] not in ('fit', 'inner_held'):
            raise ValueError('Unexpected source partition')
        for key, hashkey in [('source', 'source_sha256'), ('cache', 'cache_sha256')]:
            if digest(Path(r[key])) != r[hashkey]:
                raise ValueError('Changed source')
        cache = load_cache(Path(r['cache']))
        config = json.loads((Path(r['collection'])/'resolved_config.json').read_text())
        if digest(Path(r['collection'])/'resolved_config.json') != r['config_sha256']:
            raise ValueError('Changed robot configuration')
        fk = RobotFK(config, 'left')
        joints = {j.get('name'): j for j in ET.parse(fk.urdf).getroot().findall('joint')}
        if digest(fk.urdf) != r['urdf_sha256']:
            raise ValueError('Changed URDF')
        with h5py.File(r['source']) as h:
            commands = np.asarray(h['joint_action/vector'], np.float32)
            measured = np.asarray(h['cx_measurement/arm_qpos'], np.float32).reshape(-1, 12)
            arm = commands[:, [0,1,2,3,4,5,7,8,9,10,11,12]]
            clock = timeline(h['cx_measurement/physics_tick'][:], h['cx_dense/tick_before'][:], h['cx_dense/dt_s'][:])
            if str(h.attrs['instruction']) != r['instruction'] or int(h.attrs['seed']) != r['seed']:
                raise ValueError('Instruction/seed mismatch')
            dt = []
            for i, frame in enumerate(cache['frame_index']):
                target, valid = expert_chunk(commands, int(frame))
                state = commands[frame].copy()
                state[[0,1,2,3,4,5,7,8,9,10,11,12]] /= np.pi
                if not np.array_equal(target, cache['target'][i]) or not np.array_equal(valid, cache['valid'][i]) or not np.array_equal(state, cache['context'][i,:14]):
                    raise ValueError('Same-state command/label alignment failed')
                dt.extend(np.diff(clock['time_s'][frame:frame+1+int(valid.sum())]).tolist())
            tracked = dict(max_command_step_by_arm_channel=np.abs(np.diff(arm,axis=0)).max(0).tolist(),
                max_measured_step_by_arm_channel=np.abs(np.diff(measured,axis=0)).max(0).tolist(),
                command_tracking_abs_p99_by_arm_channel=np.quantile(np.abs(arm-measured),.99,axis=0).tolist(),
                current_to_next_command_dt_quantiles=np.quantile(dt,[0,.25,.5,.75,1]).tolist(),
                zero_dt_slots=int(np.sum(np.array(dt)==0)), command_frames=len(commands))
        residual, sq, floor, grad = error_components(cache['base_action'],cache['target'],cache['valid'])
        query_error = sq.sum((1,2))
        native = cache['target'].astype(float)-cache['base_action']
        native_valid = native[cache['valid']]
        channel_error = sq.sum(1).mean(0)
        bins = [(0,4),(4,8),(8,16),(16,32)]
        top = []
        for index in np.argsort(query_error)[-3:][::-1]:
            h,c = np.unravel_index(np.argmax(sq[index]),sq[index].shape)
            frame = int(cache['frame_index'][index]); end = min(frame+32,len(commands)-1)
            top.append(dict(frame=frame,episode_fraction=frame/(len(commands)-1),query_mse=float(query_error[index]),
                worst_horizon=int(h+1),worst_channel=int(c),base=float(cache['base_action'][index,h,c]),
                target=float(cache['target'][index,h,c]),
                left_wrist_command_trace=commands[frame:end+1,5].tolist(),
                left_wrist_measured_trace=measured[frame:end+1,5].tolist()))
        records.append(dict(task=r['task'],group=r['group'],split=r['pilot_split'],source=r['source'],
            source_sha256=r['source_sha256'],cache_sha256=r['cache_sha256'],queries=len(query_error),
            normalized_mse=float(query_error.mean()),oracle_floor=float(floor.sum((1,2)).mean()),
            channel_mse_contribution=channel_error.tolist(),
            horizon_mse_contribution=[float(sq[:,a:b].sum((1,2)).mean()) for a,b in bins],
            channel_absolute_output_gradient=grad.sum(1).mean(0).tolist(),
            beyond_cap_gradient_contribution=float((grad*(np.abs(residual)>1)).sum((1,2)).mean()),
            total_gradient_contribution=float(grad.sum((1,2)).mean()),
            native_abs_difference_quantiles=np.quantile(np.abs(native_valid),[.5,.9,.99,1],axis=0).tolist(),
            arm_difference_gt_pi=int((np.abs(native_valid[:,[0,1,2,3,4,5,7,8,9,10,11,12]])>np.pi).sum()),
            left_arm_urdf=[dict(name=n,kind=joints[n].get('type'),limits=joints[n].find('limit').attrib) for n in fk.names],
            top_queries=top,alignment='exact',**tracked))
    summary = {}
    for split in ('fit','inner_held'):
        summary[split] = {}
        for task in sorted({r['task'] for r in records}):
            rr = [r for r in records if r['split']==split and r['task']==task]
            mse = np.mean([r['normalized_mse'] for r in rr]);floor = np.mean([r['oracle_floor'] for r in rr])
            channels = np.mean([r['channel_mse_contribution'] for r in rr],axis=0)
            gradient = np.mean([r['channel_absolute_output_gradient'] for r in rr],axis=0)
            summary[split][task] = dict(episodes=len(rr),queries=sum(r['queries'] for r in rr),mse=float(mse),
                oracle_gain=1-float(floor/mse),channel_error_share=(channels/mse).tolist(),
                channel_output_gradient_share=(gradient/gradient.sum()).tolist(),
                beyond_cap_gradient_fraction=float(np.mean([r['beyond_cap_gradient_contribution'] for r in rr])/np.mean([r['total_gradient_contribution'] for r in rr])),
                horizon_error_share=(np.mean([r['horizon_mse_contribution'] for r in rr],axis=0)/mse).tolist(),
                episode_error_shares={r['group']:r['normalized_mse']/sum(q['normalized_mse'] for q in rr) for r in rr})
    write_json(run/'result.json',dict(status='complete',records=records,summary=summary,
        limits='Native command imitation, not object motion/task success. Absolute joint drives have winding/path semantics; no angle wrapping is applied. Error concentration is descriptive; no episode, channel or horizon dropped from training/evaluation.'))
    print(json.dumps(dict(output=str(run),summary=summary),indent=2),flush=True)


if __name__ == '__main__':
    main()
