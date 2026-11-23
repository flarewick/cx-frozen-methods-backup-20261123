"""Audit first compatible-state replay and original-constraint planner query."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .constraint_teacher import hold_compatible
from .paths import write_json
from .teacher_projection import CAPS

def audit(run):
    read=lambda p:json.loads(p.read_text())
    manifest=read(run/'manifest.json');folder=run/'align_capped'
    sources=manifest['source_sha256']
    trace_source=next(Path(p) for p in sources if p.endswith('/align_capped/trace.jsonl'))
    source=json.loads(trace_source.read_text().splitlines()[0])
    trace=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
    row=trace[0];fork=read(folder/'fork.json');result=read(folder/'result.json');query=read(folder/'resumed_approach.json')
    expected=np.asarray(row['base'],np.float32)+np.clip(np.asarray(row['desired'],np.float32)-np.asarray(row['base'],np.float32),-CAPS,CAPS)
    x=row['inspection'];metric=hold_compatible(x['rotation_metric'],x['position_error_m'],x['weights'],x['project_to_goal_frame'])
    checks=dict(sources=all(digest(Path(p))==v for p,v in sources.items()),code=all(digest(Path(p))==v for p,v in manifest['code_sha256'].items()),complete=read(run/'result.json')['status']=='complete',
        single_action=len(trace)==1 and result['executed']==1 and result['native_count']==117 and query['native_count']==117,
        parent_state=row['after_observation']==source['after_observation'] and row['diagnostics']['physical']==source['diagnostics']['physical'] and row['action']==source['action'],
        capped_action=np.array_equal(expected,np.asarray(row['action'],np.float32)),
        chain=fork['observation']==row['before_observation'] and fork['native_count']==116,
        native_compatible=bool(metric and x['compatible'] and row['ready']),
        unchanged_query=query['pose']==fork['inspection']['requested_world_pose'] and query['constraint']==fork['inspection']['requested_constraint'],
        honest_outcome=result['original_approach_accepted']==(query['status']=='Success') and result['censored_episode'])
    return dict(passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},rotation_metric=x['rotation_metric'],anchor_drift_m=row['anchor_drift_m'],original_approach_accepted=result['original_approach_accepted'],motion_gen=query['motion_gen'],scope='query feasibility at exact saved state, no approach execution or episode improvement')

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();report=audit(a.run.resolve());write_json(a.run.resolve()/'audit.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
