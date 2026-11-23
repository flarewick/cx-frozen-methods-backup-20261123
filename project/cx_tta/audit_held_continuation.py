"""Native held-entry continuation: saved action, source and terminal audit."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_teacher_projection import saved_signature
from .paths import write_json

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda p:json.loads(p.read_text());m=read(run/'manifest.json');r=read(run/'result.json');fork=read(run/'fork.json');entry=read(run/'stage_entry.json');support_run=Path(m['support_run']);support=read(support_run/'contact_reanalysis/result.json');support_fork=read(support_run/'hold_native/fork.json')
    checks=dict(code=all(digest(Path(p))==h for p,h in m['code_sha256'].items()),protocol=digest(Path(m['protocol']))==m['protocol_sha256'],support=support['passed'] and all(x['local_support_passed'] for x in support['reports']) and digest(support_run/'contact_reanalysis/result.json')==m['support_result_sha256'],fork=fork['physical_match']['matched'] and fork['diagnostics']['physical']==support_fork['diagnostics']['physical'] and fork['observation']==support_fork['observation']==saved_signature(run/'fork.npz'),entry=entry['actor']==support_fork['object_name'] and entry['side']=='right' and entry['native_count']==64 and entry['skipped']=='pregrasp and approach' and entry['retained']=='close_gripper',chain=True,actions=True,plans=True,accounting=True,terminal=True)
    rows=[json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()];plans={};previous=fork['observation'];count=64
    for row in rows:
        pid=row['plan']
        if pid not in plans:
            with np.load(run/f'plan_{pid:03d}.npz',allow_pickle=False) as z:plans[pid]=(z['dense_commands'].copy(),z['indices'].copy())
        dense,indices=plans[pid];idx=row['dense_index']
        checks['plans'] &= idx in indices and np.array_equal(np.asarray(row['action'],np.float32),dense[idx])
        checks['chain'] &= row['before_observation']==previous and row['action_count']==count+1 and row['physics_steps']>0
        previous=row['after_observation'];count=row['action_count']
    with np.load(run/'native_actions.npz',allow_pickle=False) as z:checks['actions'] &= np.array_equal(z['actions'],np.asarray([x['action'] for x in rows],np.float32))
    checks['accounting'] &= len(rows)==r['teacher_actions']==r['observation_updates'] and count==r['action_count']==64+len(rows) and count<=500 and sum(x['physics_steps'] for x in rows)==r['physics_steps'] and len(plans)==r['teacher_plans'] and digest(run/'trace.jsonl')==r['trace_sha256']
    final=r['final_diagnostics'];checks['terminal'] &= final['physical']==rows[-1]['offline_only']['physical'] and saved_signature(run/'terminal.npz')==previous and r['native_terminal']==bool(r['teacher_success'] or count==500)
    # Recompute exact frozen shoe success predicate. Gripper-open terms use the
    # same commanded predicate as the official task, not an aperture certificate.
    obj=np.asarray(final['physical']['offline_only']['object_poses']['shoe']);quat=obj[3:].copy()
    if quat[0]<0:quat*=-1
    recomputed=bool(np.all(np.abs(obj[:2]-[0,-.08])<[.05,.02]) and np.all(np.abs(quat-[.5,.5,-.5,-.5])<.07) and all(final['grippers_open']))
    checks['terminal'] &= recomputed==r['teacher_success'] and (r['terminal_success']==r['teacher_success'] if r['native_terminal'] else r['terminal_success'] is None)
    checks={k:bool(v) for k,v in checks.items()};report=dict(passed=all(checks.values()),checks=checks,teacher_success=r['teacher_success'],native_terminal=r['native_terminal'],continuation_actions=len(rows),native_count=count,reason=r['reason'],official_success_recomputed=recomputed,scope='single developer state privileged unbounded native continuation; no bounded residual or student/generalization improvement claim')
    write_json(run/'audit.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
