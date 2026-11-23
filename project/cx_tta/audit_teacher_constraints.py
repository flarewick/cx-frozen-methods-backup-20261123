"""Read-only-source audit of saved constraint-interface probe records."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .constraint_teacher import hold_compatible, quaternion_degrees
from .paths import write_json
from .teacher_projection import CAPS

def audit(run):
    def read(path):return json.loads(path.read_text())
    m=read(run/'manifest.json');parent=Path(m['parent'])/'projected'
    checks={'source_hashes':all(digest(Path(p))==v for p,v in m['source_sha256'].items()),'code_hashes':all(digest(Path(p))==v for p,v in m['code_sha256'].items())}
    result=read(run/'result.json');checks['complete']=result['status']=='complete' and [r['branch'] for r in result['reports']]==list(m['branches'])
    source=[json.loads(x) for x in (parent/'trace.jsonl').read_text().splitlines()]
    with np.load(parent/'base_0001.npz',allow_pickle=False) as z:base=z['actions'][20:32]
    findings=[]
    for name in m['branches']:
        folder=run/name;fork=read(folder/'fork.json');r=read(folder/'result.json')
        records=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()] if (folder/'trace.jsonl').exists() else []
        b={'fork':fork['native_count']==116 and fork['exact_source_replay'] and fork['diagnostics']['physical']==source[51]['offline_only']['physical'] and fork['observation']==source[51]['after_observation'],
           'budget':len(records)==r['executed']<=12 and r['native_count']==116+len(records) and r['censored_episode'], 'chain':True,'actions':True,'metrics':True,'readiness':True}
        inspections=[fork['inspection']]+[x['inspection'] for x in records]+[r['final_inspection']]
        for x in inspections:
            compatible=not x['active'] or hold_compatible(x['rotation_metric'],x['position_error_m'],x['weights'],x['project_to_goal_frame'])
            degrees=quaternion_degrees(np.asarray(x['measured_planner_pose']['quaternion']).reshape(4),np.asarray(x['goal_planner_pose']['quaternion']).reshape(4))
            b['metrics'] &= compatible==x['compatible'] and abs(degrees-x['rotation_geodesic_degrees'])<1e-6
        previous=fork['observation'];anchor=np.asarray(fork['inspection']['measured_native_ee_pose'][:3])
        desired=None
        if name.startswith('align') and (folder/'alignment_commands.npz').exists():
            with np.load(folder/'alignment_commands.npz',allow_pickle=False) as z:
                desired=z['desired'].copy();b['actions'] &= np.array_equal(desired,z['dense'][z['indices']])
            g=read(folder/'alignment_goal.json');b['alignment_goal']=np.array_equal(np.asarray(g['pose'][:3]),anchor) and quaternion_degrees(g['pose'][3:],g['original_goal'][3:])<1e-5 and g['original_constraint']==[1,1,1,0,0,0]
        for i,x in enumerate(records):
            b['chain'] &= x['before_observation']==previous and x['native_count']==117+i and x['slot']==20+i;previous=x['after_observation']
            target=base[i] if desired is None else desired[min(i,len(desired)-1)]
            expected=base[i]+np.clip(target-base[i],-CAPS,CAPS) if name=='align_capped' else target
            b['actions'] &= np.array_equal(np.asarray(x['base'],np.float32),base[i]) and np.array_equal(np.asarray(x['desired'],np.float32),target) and np.array_equal(np.asarray(x['action'],np.float32),expected)
            drift=float(np.linalg.norm(np.asarray(x['inspection']['measured_native_ee_pose'][:3])-anchor))
            b['readiness'] &= abs(drift-x['anchor_drift_m'])<1e-10 and x['ready']==bool(x['inspection']['compatible'] and drift<=.005)
            if name=='base_tail':b['control']=b.get('control',True) and x['control_matches'] and x['after_observation']==source[52+i]['after_observation'] and x['diagnostics']['physical']==source[52+i]['offline_only']['physical']
        if name=='rejection':
            q=read(folder/'original_rejection.json');b['rejected']=not fork['inspection']['compatible'] and not records and q['status']=='Fail' and q['motion_gen']['calls'][0]['status']=='MotionGenStatus.INVALID_PARTIAL_POSE_COST_METRIC'
        if r['reason']=='measured_ready':
            q=read(folder/'resumed_approach.json');b['resumed']=bool(records[-1]['ready']) and q['constraint']==fork['inspection']['requested_constraint'] and q['pose']==fork['inspection']['requested_world_pose'] and r['original_approach_accepted']==(q['status']=='Success')
        findings.append(dict(branch=name,passed=bool(all(b.values())),checks={k:bool(v) for k,v in b.items()},executed=r['executed'],reason=r['reason'],initial_rotation_metric=fork['inspection']['rotation_metric'],final_rotation_metric=r['final_inspection']['rotation_metric'],final_anchor_drift_m=records[-1]['anchor_drift_m'] if records else 0,original_approach_accepted=r['original_approach_accepted']))
    checks['branches']=all(x['passed'] for x in findings)
    return dict(passed=all(checks.values()),checks=checks,branches=findings,scope='saved-record audit, not independent GPU recomputation of native orientation metric or task success evaluation')

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);args=p.parse_args();report=audit(args.run.resolve());write_json(args.run.resolve()/'audit.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
