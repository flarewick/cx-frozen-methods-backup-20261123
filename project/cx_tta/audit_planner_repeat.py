"""Independent saved-array/input audit for native planner repetition."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix,error
from .audit_place_goal import BOUNDED,read
from .paths import write_json
from .robot_kinematics import RobotFK

def compare(a,b):
    n=min(len(a),len(b));dif=np.abs(a[:n].astype(float)-b[:n].astype(float));changed=np.where(np.any(dif!=0,axis=1))[0]
    return dict(exact=bool(a.shape==b.shape and np.array_equal(a,b)),lengths=[len(a),len(b)],common_prefix_max_abs=float(dif.max()),first_changed_row=None if not len(changed) else int(changed[0]))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();manifest=read(run/'manifest.json');result=read(run/'result.json')
    if result['status']!='complete' or len(result['records'])!=18:raise ValueError('18 complete calls required')
    for file,sha in manifest['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    config=read(Path(manifest['source'])/'resolved_config.json');fk=RobotFK(config,'right');reports=[]
    for name,plan in [('restart_64',BOUNDED/'compatible/plan_0000.json'),('held_preplacement_94',BOUNDED/'held_compatible/plan_0002.json')]:
        expected=read(plan);rows=[r for r in result['records'] if r['checkpoint']==name]
        if len(rows)!=9 or len({r['rng_before'] for r in rows})!=1:raise ValueError('Repeat RNG inputs not identical')
        forks=[read(run/name/f'scene_{s}/fork.json') for s in range(3)]
        if not all(x['physical']==expected['physical'] and x['observation']==expected['observation'] and x['native_count']==expected['native_count'] for x in forks):raise ValueError('Physical/observation start mismatch')
        arrays={};endpoint=[]
        for r in rows:
            if r['goal']!=expected['goal']['pose'] or r['constraint']!=expected['goal']['constraint'] or not r['physical_unchanged']:raise ValueError('Query/accounting mismatch')
            if r['status']!='Success':continue
            path=Path(r['file'])
            if digest(path)!=r['sha256']:raise ValueError('Plan changed')
            with np.load(path) as z:data={k:z[k].copy() for k in z.files}
            if not np.isfinite(data['dense']).all() or not np.array_equal(data['commands'],data['dense'][data['indices']]):raise ValueError('Exported command mismatch')
            arrays[(r['scene'],r['repeat'])]=data
            endpoint.append(dict(scene=r['scene'],repeat=r['repeat'],**error(fk.pose(data['commands'][-1,7:13]),pose_matrix(r['goal']))))
        comparisons=[]
        for first,second in itertools.combinations(sorted(arrays),2):
            if first[0]!=second[0] and (first[1]!=0 or second[1]!=0):continue
            comparisons.append(dict(first=list(first),second=list(second),kind='same_instance' if first[0]==second[0] else 'fresh_instance',
                dense=compare(arrays[first]['dense'],arrays[second]['dense']),commands=compare(arrays[first]['commands'],arrays[second]['commands'])))
        reports.append(dict(checkpoint=name,calls=len(rows),success_plans=sum(r['status']=='Success' for r in rows),start_inputs_exact=True,dense_lengths=[r.get('dense_length') for r in rows],comparisons=comparisons,endpoints=endpoint,
            identical_same_instance=all(r['commands']['exact'] for r in comparisons if r['kind']=='same_instance'),
            identical_fresh_instance=all(r['commands']['exact'] for r in comparisons if r['kind']=='fresh_instance')))
    report=dict(passed=True,reports=reports,source_sha256={str(f):digest(f) for f in [run/'manifest.json',run/'result.json',Path(__file__),fk.urdf]},
        interpretation='Exact global RNG/physical/goal inputs do not reset native planner internal state. Different plans therefore establish native internal/numerical variation under this protocol, not VLA noise. This probe does not isolate which native subsystem causes it or quantify complete-episode flip frequency.',scope='Planning repeatability only; states 64/94 are censored, no task success denominator.')
    write_json(run/'audit.json',report)
    print(json.dumps(dict(passed=True,reports=[{k:v for k,v in r.items() if k not in ('comparisons','endpoints')} for r in reports]),indent=2))

if __name__=='__main__':main()
