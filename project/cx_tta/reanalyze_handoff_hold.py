"""Correct raw contact membership without rewriting recorded GPU evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import pose_matrix,error
from .handoff_support import matching_links
from .paths import write_json
from .robot_kinematics import RobotFK
from .teacher_projection import CAPS

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda p:json.loads(p.read_text());m=read(run/'manifest.json');snap=read(run/'source_snapshot.json');raw_audit=read(run/'audit.json')
    parent=Path(m['parent']);reference=read(parent/'base/fork.json');fk=RobotFK(read(Path(m['source'])/'resolved_config.json'),'right')
    with np.load(parent/'common_base.npz',allow_pickle=False) as z:base=z['base'].copy()
    checks=dict(source_hashes=all(digest(Path(p))==h for p,h in m['source_sha256'].items()),executed_code_snapshots=all(digest(Path(x['snapshot']))==x['sha256']==m['code_sha256'][x['source']] for x in snap),raw_failure_retained=not raw_audit['passed'],complete=read(run/'result.json')['status']=='complete',branches=True)
    reports=[]
    for name in m['branches']:
        folder=run/name;fork=read(folder/'fork.json');rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];raw=read(folder/'result.json');initial=pose_matrix(fork['measured_ee_pose']);obj0=np.asarray(fork['object_matrix']);rel0=np.asarray(fork['relative_matrix']);target=initial.copy()
        if name!='hold_native':target[2,3]+=.01
        b=dict(fork=fork['observation']==reference['observation'] and fork['diagnostics']['physical']==reference['diagnostics']['physical'],chain=True,geometry=True,action=True,budget=len(rows)==12 and raw['executed']==12 and raw['native_count']==76,original_evidence=digest(folder/'trace.jsonl')==raw['trace_sha256']);contacts=[];slips=[];previous=fork['observation']
        if name=='hold_native':desired=np.tile(np.asarray(fork['current_command'],np.float32),(12,1))
        else:
            with np.load(folder/'plan.npz',allow_pickle=False) as z:
                path=z['dense'][z['indices']];desired=np.stack([path[min(i,len(path)-1)] for i in range(12)]);b['action'] &= np.array_equal(desired,z['desired'])
        for i,x in enumerate(rows):
            ee=np.asarray(x['measured_ee_matrix']);obj=np.asarray(x['object_matrix']);rel=np.linalg.inv(ee)@obj;slip=error(rel,rel0);arrival=error(ee,target)
            links=matching_links(x['diagnostics']['contacts'],fork['object_name'],fork['finger_links']);contacts.append(links);slips.append(slip)
            b['geometry'] &= np.allclose(ee,fk.pose(x['diagnostics']['physical']['arm_qpos'][1]),rtol=0,atol=1e-5) and np.allclose(obj,pose_matrix(x['diagnostics']['physical']['offline_only']['object_poses']['shoe']),rtol=0,atol=1e-12) and np.allclose(rel,np.asarray(x['relative_matrix']),rtol=0,atol=1e-12)
            b['geometry'] &= abs(slip['distance_m']-x['relative_error']['distance_m'])<1e-12 and abs(slip['angle_deg']-x['relative_error']['angle_deg'])<1e-9 and abs(arrival['distance_m']-x['target_error']['distance_m'])<1e-12 and abs(arrival['angle_deg']-x['target_error']['angle_deg'])<1e-9
            expected=base[i]+np.clip(desired[i]-base[i],-CAPS,CAPS) if name=='lift_capped' else desired[i]
            b['action'] &= np.array_equal(np.asarray(x['action'],np.float32),expected) and np.array_equal(np.asarray(x['base'],np.float32),base[i])
            b['chain'] &= x['before_observation']==previous and x['native_count']==65+i and x['physics_steps']>0;previous=x['after_observation']
        last=rows[-1];arrival=error(np.asarray(last['measured_ee_matrix']),target);ee_rise=last['measured_ee_matrix'][2][3]-initial[2,3];obj_rise=last['object_matrix'][2][3]-obj0[2,3]
        coupled=all(s['distance_m']<=.003 and s['angle_deg']<=3. and len(c)==len(fork['finger_links']) for s,c in zip(slips,contacts));arrived=arrival['distance_m']<=.005 and arrival['angle_deg']<=3.;passed=bool(coupled and arrived and (name=='hold_native' or ee_rise>=.008 and obj_rise>=.005))
        b={k:bool(v) for k,v in b.items()};checks['branches'] &= all(b.values());reports.append(dict(branch=name,record_verification_passed=all(b.values()),checks=b,local_support_passed=passed,coupling_passed=bool(coupled),arrival_passed=bool(arrived),both_finger_contact_actions=sum(len(c)==len(fork['finger_links']) for c in contacts),final_ee_rise_m=float(ee_rise),final_object_rise_m=float(obj_rise),max_relative_drift_m=max(s['distance_m'] for s in slips),max_relative_angle_deg=max(s['angle_deg'] for s in slips),raw_contact_classifier_correct=False))
    folder=run/'contact_reanalysis';folder.mkdir();checks={k:bool(v) for k,v in checks.items()}
    write_json(folder/'result.json',dict(status='complete',passed=all(checks.values()),checks=checks,reports=reports,raw_audit_sha256=digest(run/'audit.json'),source_result_sha256=digest(run/'result.json'),code_sha256={str(f):digest(f) for f in [Path(__file__),Path(__file__).with_name('handoff_support.py')]},defect='runtime tuple contacts compared with list key; decisions never affect probe action execution',scope='recomputed local support verdict; raw traces/result/failed audit preserved; no full recovery or force-closure certificate'))
    print(json.dumps(dict(passed=all(checks.values()),checks=checks,reports=reports),indent=2));raise SystemExit(0 if all(checks.values()) else 1)
if __name__=='__main__':main()
