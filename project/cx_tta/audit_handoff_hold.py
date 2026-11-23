"""Recompute saved local holding/lifting geometry and action/budget contract."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_local_geometry import error
from .paths import write_json
from .teacher_projection import CAPS

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda p:json.loads(p.read_text());m=read(run/'manifest.json');result=read(run/'result.json');parent=Path(m['parent']);ref=read(parent/'base/fork.json')
    with np.load(parent/'common_base.npz',allow_pickle=False) as z:base=z['base']
    checks=dict(source_hashes=all(digest(Path(p))==h for p,h in m['source_sha256'].items()),code_hashes=all(digest(Path(p))==h for p,h in m['code_sha256'].items()),complete=result['status']=='complete' and len(result['reports'])==3)
    reports=[]
    for name in m['branches']:
        folder=run/name;fork=read(folder/'fork.json');r=read(folder/'result.json');rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];b=dict(fork=fork['observation']==ref['observation'] and fork['diagnostics']['physical']==ref['diagnostics']['physical'],budget=len(rows)==12 and r['executed']==12 and r['native_count']==76 and r['censored_episode'],chain=True,actions=True,geometry=True,contacts=True,verdict=True)
        previous=fork['observation'];relative0=np.asarray(fork['relative_matrix']);object0=np.asarray(fork['object_matrix']);ee0=np.asarray(fork['measured_ee_pose'][:3]);coupled=True
        current=np.asarray(fork['current_command'],np.float32)
        if name!='hold_native':
            with np.load(folder/'plan.npz',allow_pickle=False) as z:
                commands=z['dense'][z['indices']];desired=np.stack([commands[min(i,len(commands)-1)] for i in range(12)])
                b['actions'] &= np.array_equal(desired,z['desired'])
        else:desired=np.tile(current,(12,1))
        for i,x in enumerate(rows):
            ee,obj,relative=map(np.asarray,[x['measured_ee_matrix'],x['object_matrix'],x['relative_matrix']]);slip=error(np.linalg.inv(ee)@obj,relative0)
            b['geometry'] &= np.allclose(relative,np.linalg.inv(ee)@obj,rtol=0,atol=1e-12) and abs(slip['distance_m']-x['relative_error']['distance_m'])<1e-12 and abs(slip['angle_deg']-x['relative_error']['angle_deg'])<1e-9 and abs(ee[2,3]-ee0[2]-x['ee_rise_m'])<1e-12 and abs(obj[2,3]-object0[2,3]-x['object_rise_m'])<1e-12
            contacts=[link for link in fork['finger_links'] if sorted([fork['object_name'],link]) in x['diagnostics']['contacts']]
            b['contacts'] &= contacts==x['finger_contacts'];coupled &= slip['distance_m']<=.003 and slip['angle_deg']<=3. and len(contacts)==len(fork['finger_links'])
            expected=base[i]+np.clip(desired[i]-base[i],-CAPS,CAPS) if name=='lift_capped' else desired[i]
            b['actions'] &= np.array_equal(np.asarray(x['base'],np.float32),base[i]) and np.array_equal(np.asarray(x['desired'],np.float32),desired[i]) and np.array_equal(np.asarray(x['action'],np.float32),expected)
            b['chain'] &= x['native_count']==65+i and x['before_observation']==previous and x['physics_steps']>0;previous=x['after_observation']
        final=rows[-1];arrival=final['target_error']['distance_m']<=.005 and final['target_error']['angle_deg']<=3.;passed=bool(coupled and arrival and (name=='hold_native' or final['ee_rise_m']>=.008 and final['object_rise_m']>=.005))
        b['verdict'] &= r['coupling_passed']==bool(coupled) and r['arrival_passed']==arrival and r['local_support_passed']==passed and r['trace_sha256']==digest(folder/'trace.jsonl')
        reports.append(dict(branch=name,passed=all(b.values()),checks={k:bool(v) for k,v in b.items()},local_support_passed=passed,max_relative_drift_m=r['max_relative_drift_m'],final_ee_rise_m=final['ee_rise_m'],final_object_rise_m=final['object_rise_m']))
    checks['branches']=all(r['passed'] for r in reports);report=dict(passed=all(checks.values()),checks=checks,branches=reports,scope='saved geometry/action verification; local support criteria do not certify force closure or full task recovery');write_json(run/'audit.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
