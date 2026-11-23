"""S1h: source-backed relative geometry and bounded endpoint FK diagnostics.

Privileged offline analysis only. No simulation, policy query, training label or
claim of collision/path/dynamic execution feasibility is produced.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .audit import digest
from .paths import ROOT,new_run,write_json
from .robot_kinematics import RobotFK

DEFAULT=ROOT/'outputs/compatible_continuations/place_shoe/gate_20261001_141842_178427'
COUNTS=(100,110,143)
ANGLE_LIMIT=np.deg2rad(3.) # original predeclared move-arrival orientation, NOT native .05 metric

def pose_matrix(pose):
    p=np.asarray(pose,float)
    if p.shape!=(7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:])<1e-12:raise ValueError('Invalid pose')
    out=np.eye(4);out[:3,3]=p[:3];out[:3,:3]=Rotation.from_quat(p[3:][[1,2,3,0]]).as_matrix();return out

def error(actual,target):
    return dict(distance_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),angle_deg=float(np.degrees(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())))

def joint_bounds(fk):
    elements={j.get('name'):j for j in ET.parse(fk.urdf).getroot().findall('joint')}
    limits=[]
    for name in fk.names:
        joint=elements[name];limit=joint.find('limit')
        if joint.get('type')=='continuous':limits.append((-np.inf,np.inf))
        elif limit is None:raise ValueError('Missing joint limits: '+name)
        else:limits.append((float(limit.get('lower')),float(limit.get('upper'))))
    return np.asarray(limits)

def optimize_endpoint(fk,current,base,desired,target,limits):
    lo=np.maximum(base-.1,limits[:,0]);hi=np.minimum(base+.1,limits[:,1])
    if np.any(lo>=hi):return dict(status='empty_joint_box',lower=lo.tolist(),upper=hi.tolist())
    def orientation(q):return Rotation.from_matrix(fk.pose(q)[:3,:3]@target[:3,:3].T).magnitude()
    def cost(q):return np.sum((fk.pose(q)[:3,3]-target[:3,3])**2)/(.01**2)+1e-5*np.sum((q-current)**2)
    # Finite differences must resolve a float64 CPU FK, no float32 GPU FD mismatch.
    constraint={'type':'ineq','fun':lambda q:ANGLE_LIMIT**2-orientation(q)**2}
    rng=np.random.default_rng(910001)
    starts=[np.clip(base,lo,hi),np.clip(desired,lo,hi),np.clip(current,lo,hi)]+[rng.uniform(lo,hi) for _ in range(5)]
    trials=[]
    for seed,x in enumerate(starts):
        fit=minimize(cost,x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=constraint,
            options=dict(maxiter=150,ftol=1e-10,eps=1e-6))
        q=fit.x;value=error(fk.pose(q),target)
        feasible=bool(np.all(q>=lo-1e-8) and np.all(q<=hi+1e-8) and value['angle_deg']<=3.+1e-5)
        trials.append(dict(seed=seed,solver_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),endpoint_orientation_feasible=feasible,qpos=q.tolist(),**value))
    feasible=[x for x in trials if x['endpoint_orientation_feasible']]
    best=min(feasible,key=lambda x:x['distance_m']) if feasible else None
    return dict(status='feasible_endpoint_found' if best else 'no_feasible_endpoint_found_by_local_search',best=best,starts=len(starts),trials=trials,lower=lo.tolist(),upper=hi.tolist(),orientation_limit_deg=3.,not_checked=['collision','path','realized dynamics','gripper-object coordination','task success'])

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--source',type=Path,default=DEFAULT);args=parser.parse_args();source=args.source.resolve()
    read=lambda p:json.loads(p.read_text())
    if not read(source/'audit.json')['passed']:raise ValueError('Source audit failed')
    manifest=read(source/'manifest.json');folder=source/'compatible';config_path=Path(manifest['source'])/'resolved_config.json'
    protocol=read(Path(manifest['protocol']));point=protocol['points'][manifest['point']]
    if digest(config_path)!=point['source_config_sha256']:raise ValueError('Source configuration changed')
    for name,sha in read(source/'analysis.json')['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed source artifact')
    rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
    measurements=[json.loads(x) for x in (folder/'measurements.jsonl').read_text().splitlines()]
    fork=read(folder/'fork.json');states={64:fork['diagnostics']['physical']};states.update({x['count']:x['offline_only']['physical'] for x in rows if x['advanced']})
    fk=RobotFK(read(config_path),'right');limits=joint_bounds(fk)
    # URDF control link is compared directly to native measured EE-reference.
    # No fitted frame correction or per-checkpoint transform is applied.
    fk_checks=[]
    for x in measurements:
        if x['kind']=='move':fk_checks.append(error(fk.pose(states[x['native_count']]['arm_qpos'][1]),pose_matrix(x['arrival']['pose'])))
    max_distance=max(x['distance_m'] for x in fk_checks);max_angle=max(x['angle_deg'] for x in fk_checks)
    if max_distance>1e-5 or max_angle>.001:raise ValueError('CPU/native measured frame parity failed')
    phase1=[x for x in measurements if x['phase']==1 and x['kind']=='move'];goal=pose_matrix(phase1[0]['goal'])
    object0=pose_matrix(states[64]['offline_only']['object_poses']['shoe']);local_goal=np.linalg.inv(object0)@goal
    run=new_run('local_geometry',group=('reports','analyses'));print(run,flush=True)
    write_json(run/'manifest.json',dict(source=str(source),source_sha256={str(p):digest(p) for p in [source/'audit.json',folder/'trace.jsonl',folder/'measurements.jsonl',folder/'fork.json',config_path,fk.urdf]},code_sha256={str(p):digest(p) for p in [Path(__file__),ROOT/'cx_tta/robot_kinematics.py']},counts=COUNTS,optimizer='8 fixed starts SLSQP;150 iterations;0.1 residual box intersected with URDF limits;3deg endpoint orientation',goal_transport='fixed local original contact frame transported with object pose, no contact reselection; privileged diagnostic, not an official reselected goal',scope='CPU offline endpoint geometry; no executed candidates or recovery labels'))
    results=[]
    for count in COUNTS:
        state=states[count];nextrow=next(x for x in rows if x['count']==count+1 and x['advanced'])
        current=np.asarray(state['arm_qpos'][1]);base=np.asarray(nextrow['base'][7:13]);desired=np.asarray(nextrow['desired'][7:13]);action=np.asarray(nextrow['action'][7:13])
        actual=fk.pose(current);obj=pose_matrix(state['offline_only']['object_poses']['shoe']);transported=obj@local_goal
        checks=dict(native_count=count,next_count=count+1,mode=nextrow['mode'],current_qpos=current.tolist(),base=base.tolist(),desired=desired.tolist(),clipped_action=action.tolist(),object_motion=error(obj,object0),target_transport_delta=error(transported,goal),goals={})
        for name,target in [('fixed',goal),('object_transport',transported)]:
            values={k:error(fk.pose(q),target) for k,q in [('current',current),('base',base),('desired',desired),('clipped',action)]}
            optimized=optimize_endpoint(fk,current,base,desired,target,limits)
            checks['goals'][name]=dict(target_matrix=target.tolist(),endpoints=values,optimized=optimized)
        results.append(checks);write_json(run/f'count_{count}.json',checks)
        print('CHECKPOINT',count,[(name,x['optimized']['status']) for name,x in checks['goals'].items()],flush=True)
    write_json(run/'result.json',dict(status='complete',fk_parity=dict(poses=len(fk_checks),max_distance_m=max_distance,max_angle_deg=max_angle,frame_correction='none'),records=results,scope='kinematic endpoint search only; no global infeasibility or recovery/success claim'))
    print('COMPLETE',run,flush=True)

if __name__=='__main__':main()
