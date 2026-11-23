"""Recompute support verdicts from retained fork, geometry and native traces."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix,error
from .cache_recovery import verify_hashes
from .paths import write_json

def read(p):return json.loads(p.read_text())

def audit(run):
    m=read(run/'manifest.json');r=read(run/'result.json');verify_hashes(m['code_sha256'])
    protocol=Path(m['protocol']);p=read(protocol)
    if digest(protocol)!=m['protocol_sha256']:raise ValueError('Protocol changed')
    point=p['points'][m['point']];sources=[run/'manifest.json',run/'result.json',protocol];reports=[]
    for report in r['reports']:
        folder=run/report['branch'];fork=read(folder/'fork.json');sources.append(folder/'fork.json')
        if fork['observation']!=point['observation_signature'] or not fork['physical_match']['matched']:raise ValueError('Wrong fork')
        ids=fork['finger_entity_ids']
        if len(set(ids['left']+ids['right']))!=4:raise ValueError('Ambiguous finger instance identities')
        owners=[s for s in ('left','right') if len(fork['finger_contacts'][s])==2 and all(fork['finger_contacts'][s])]
        if owners!=fork['candidate_owners']:raise ValueError('Owner classification changed')
        if report.get('reason')=='no_unique_two_finger_owner':
            if len(owners)==1 or report['passed']:raise ValueError('Wrong no-owner verdict')
            reports.append(dict(branch=report['branch'],passed=False,reason=report['reason']));continue
        side=report['owner']
        if owners!=[side]:raise ValueError('Wrong side')
        if report.get('reason')=='lift_planning_failed':
            if read(folder/'planning.json')['status']=='Success' or report['passed']:raise ValueError('Wrong planning verdict')
            reports.append(dict(branch=report['branch'],passed=False,reason=report['reason']));continue
        rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];sources.append(folder/'trace.jsonl')
        ee0=pose_matrix(fork['ee_pose_by_side'][side]);obj0=pose_matrix(fork['diagnostics']['physical']['offline_only']['object_poses']['shoe']);rel0=np.linalg.inv(ee0)@obj0
        target=ee0.copy();target[2,3]+=.01 if report['branch']=='lift' else 0
        previous=fork['observation'];coupled=len(rows)==12
        for i,row in enumerate(rows):
            if row['before_observation']!=previous or row['native_count']!=point['prefix_actions']+i+1 or row['physics_steps']<=0:raise ValueError('Native chain')
            previous=row['after_observation'];ee=np.array(row['ee']);obj=np.array(row['object']);slip=error(np.linalg.inv(ee)@obj,rel0);arrival=error(ee,target)
            for actual,expected in [(row['relative_error'],slip),(row['target_error'],arrival)]:
                if abs(actual['distance_m']-expected['distance_m'])>1e-10 or abs(actual['angle_deg']-expected['angle_deg'])>1e-6:raise ValueError('Geometry mismatch')
            if abs(row['ee_rise']-(ee[2,3]-ee0[2,3]))>1e-10 or abs(row['object_rise']-(obj[2,3]-obj0[2,3]))>1e-10:raise ValueError('Wrong lift')
            coupled &= slip['distance_m']<=.003 and slip['angle_deg']<=3 and all(row['finger_contacts'][side])
        last=rows[-1] if rows else None
        arrived=bool(last and last['target_error']['distance_m']<=.005 and last['target_error']['angle_deg']<=3)
        passed=bool(coupled and arrived and (report['branch']=='hold' or last['ee_rise']>=.008 and last['object_rise']>=.005))
        if passed!=report['passed'] or bool(coupled)!=report['coupled'] or arrived!=report['arrival']:raise ValueError('Support verdict mismatch')
        reports.append(dict(branch=report['branch'],passed=passed,owner=side))
    qualified=len(reports)==2 and all(x['passed'] for x in reports) and len({x['owner'] for x in reports})==1
    if qualified!=r['held_qualified']:raise ValueError('Combined support verdict mismatch')
    result=dict(passed=True,held_qualified=qualified,point=m['point'],reports=reports,
        source_sha256={str(f):digest(f) for f in sources},auditor_sha256=digest(Path(__file__)),
        scope='Native-chain/recorded support reanalysis, not force closure, teacher completion or adapter evaluation')
    write_json(run/'audit.json',result);print(json.dumps(result),flush=True);return result

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();audit(a.run.resolve())

if __name__=='__main__':main()
