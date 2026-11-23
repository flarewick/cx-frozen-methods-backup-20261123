"""Source-backed held-transform placement audit; privileged diagnostics only."""
import argparse
import ast
import __future__
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import transforms3d as t3d
from .audit import digest
from .audit_local_geometry import error, pose_matrix
from .paths import ROOT, new_run, write_json
from .robot_kinematics import RobotFK

BOUNDED=ROOT/'outputs/held_bounded_continuations/place_shoe/gate_20261002_035042_869420'
NATIVE=ROOT/'outputs/recovery_stage_controls/place_shoe/probe_20261001_155532_413918'

def read(path):return json.loads(path.read_text())

class Pose:
    def __init__(self,value):self.matrix=np.asarray(value,float).copy() if np.shape(value)==(4,4) else pose_matrix(value)
    @property
    def p(self):return self.matrix[:3,3]
    def to_transformation_matrix(self):return self.matrix.copy()

def to_pose(value):return value if isinstance(value,Pose) else Pose(value)

def matrix_pose(matrix):return np.r_[matrix[:3,3],t3d.quaternions.mat2quat(matrix[:3,:3])].tolist()

def official_math(runtime):
    """Execute only exact pure AST bodies, never import frozen simulator modules."""
    alignments=[]
    env=dict(np=np,t3d=t3d,_toPose=to_pose,transforms=SimpleNamespace(_toPose=to_pose))
    source=runtime/'envs/utils/transforms.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('get_align_matrix','get_product_vector','get_place_pose')]
    if len(nodes)!=3:raise ValueError('Official pure helpers missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),env)
    original=env['get_align_matrix']
    def tracked(a,b):
        a,b=np.asarray(a).reshape(3),np.asarray(b).reshape(3);a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
        alignments.append(dict(dot=float(a@b),cross_norm=float(np.linalg.norm(np.cross(a,b))),antiparallel_singularity=bool(np.linalg.norm(np.cross(a,b))<1e-6 and a@b<0)))
        return original(a,b)
    env['get_align_matrix']=tracked
    taskfile=runtime/'envs/_base_task.py'
    cls=next(n for n in ast.parse(taskfile.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Base_Task')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_place_pose')
    node.name='official_task_place_goal'
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(taskfile),'exec',flags=__future__.annotations.compiler_flag),env)
    return env['official_task_place_goal'],alignments

def reconstruct(runtime,state,hand,config,pre_dis=.12):
    obj=pose_matrix(state['offline_only']['object_poses']['shoe'])
    block=pose_matrix(state['offline_only']['object_poses']['target_block'])
    targetlocal=np.array([[0,-1,0,0],[-1,0,0,0],[0,0,-1,0],[0,0,0,1]],float)
    local=np.asarray(config['functional_matrix'][0],float).copy();local[:3,3]*=np.asarray(config['scale'])
    target=block@targetlocal;functional=obj@local
    actor=SimpleNamespace(get_pose=lambda:Pose(obj),get_functional_point=lambda *args:Pose(functional))
    robot=SimpleNamespace(get_right_ee_pose=lambda:matrix_pose(hand))
    fake=SimpleNamespace(plan_success=True,robot=robot)
    function,alignments=official_math(runtime)
    goal=pose_matrix(function(fake,actor,'right',matrix_pose(target),constrain='align',functional_point_id=0,pre_dis=pre_dis))
    held=np.linalg.inv(hand)@obj;goal_object=goal@held
    expected=np.eye(4);expected[:3,:3]=target[:3,:3]@local[:3,:3].T
    expected[:3,3]=target[:3,3]-expected[:3,:3]@local[:3,3]
    # Independent held-transform formula, with an explicit preplacement clearance.
    independent=expected@np.linalg.inv(held)
    bias=expected[:3,3]-independent[:3,3]
    independent[:3,3]-=pre_dis*bias/np.linalg.norm(bias)
    canonical=pose_matrix([0,0,0,.5,.5,-.5,-.5])
    return dict(goal=matrix_pose(goal),held_matrix=held.tolist(),predicted_object_pose=matrix_pose(goal_object),
        independent_formula_error=error(goal,independent),object_goal_orientation_error_deg=error(goal_object,expected)['angle_deg'],
        canonical_orientation_error_deg=error(goal_object,canonical)['angle_deg'],
        current_to_goal_object_angle_deg=error(obj,goal_object)['angle_deg'],alignment_calls=alignments,
        current_sole_down_axis=(-obj[:3,1]).tolist(),current_toe_axis=obj[:3,2].tolist(),
        goal_sole_down_axis=(-goal_object[:3,1]).tolist(),goal_toe_axis=goal_object[:3,2].tolist())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--planning-run',type=Path);a=p.parse_args()
    if not read(BOUNDED/'audit.json')['passed'] or not read(NATIVE/'audit.json')['passed']:raise ValueError('Source audits failed')
    source=Path(read(BOUNDED/'manifest.json')['source']);runtime=source/'robotwin';configpath=source/'resolved_config.json';fk=RobotFK(read(configpath),'right')
    planpath=BOUNDED/'held_compatible/plan_0002.json';plan=read(planpath)
    tracepath=NATIVE/'trace.jsonl';rows=[json.loads(x) for x in tracepath.read_text().splitlines()]
    native95=next(x['offline_only']['physical'] for x in rows if x['action_count']==95)
    checkpoints=[('bounded_preplacement_94',plan['physical'],plan['goal']['pose']),('native_preplacement_95',native95,None)]
    actual=None;inputs=[configpath,planpath,tracepath,NATIVE/'audit.json',BOUNDED/'audit.json',runtime/'envs/utils/transforms.py',runtime/'envs/_base_task.py',runtime/'envs/place_shoe.py',fk.urdf,Path(__file__)]
    if a.planning_run:
        capture=a.planning_run/'actor_metadata.json';actual=read(capture);inputs.append(capture)
        if actual['scene_seed']!=320006:raise ValueError('Actor from wrong source')
    reports=[]
    for model in range(10):
        path=runtime/f'assets/objects/041_shoe/model_data{model}.json';inputs.append(path);cfg=read(path)
        for name,state,saved in checkpoints:
            hand=fk.pose(state['arm_qpos'][1]);report=reconstruct(runtime,state,hand,cfg)
            report.update(checkpoint=name,model_id=model,actor_selection='all variants checked, no fitting/selecting a model from target error')
            if saved:report['saved_goal_orientation_error_deg']=error(pose_matrix(saved),pose_matrix(report['goal']))['angle_deg']
            if actual and model==actual['model_id']:
                report['model_confirmed_by_scene']=True
                if cfg!=actual['config']:raise ValueError('Scene actor config differs from pinned asset')
                if saved:report['saved_goal_error']=error(pose_matrix(saved),pose_matrix(report['goal']))
                if name.startswith('native'):
                    with np.load(NATIVE/'plan_002.npz') as z:command=z['dense_commands'][z['indices'][-1]]
                    report['native_plan_endpoint_error']=error(fk.pose(command[7:13]),pose_matrix(report['goal']))
            reports.append(report)
    passed=all(r['canonical_orientation_error_deg']<.001 and r['object_goal_orientation_error_deg']<.001 and r['independent_formula_error']['angle_deg']<.001 and r['independent_formula_error']['distance_m']<1e-5 and not any(x['antiparallel_singularity'] for x in r['alignment_calls']) for r in reports)
    if actual:passed=passed and all(r.get('saved_goal_error',dict(distance_m=0,angle_deg=0))['distance_m']<1e-5 for r in reports)
    run=new_run('placement_target',group=('reports','analyses'));print(run,flush=True)
    write_json(run/'result.json',dict(passed=passed,position_audit_complete=actual is not None,actual_model_id=None if actual is None else actual['model_id'],records=reports,
        source_sha256={str(f):digest(f) for f in inputs},scope='Privileged target/frame audit, no student input or recovery claim. Quaternion sign is equivalent; a 180-degree toe/heel heading change is not a permitted object symmetry in the official predicate. No modification to the official antiparallel singularity.'))
    print(json.dumps(dict(passed=passed,position_audit_complete=actual is not None,actual_model_id=None if actual is None else actual['model_id'],confirmed=[r for r in reports if r.get('model_confirmed_by_scene')],max_saved_rotation_error_deg=max(r.get('saved_goal_orientation_error_deg',0) for r in reports)),indent=2))
    if not passed:raise ValueError('Placement target gate failed')

if __name__=='__main__':main()
