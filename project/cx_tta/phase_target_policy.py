"""Offline bounded phase-completion probes; expert goals are labels, never NN inputs."""
import numpy as np
from .structured_action_adapter import ARM_INDICES,FORWARD_KEYS
from .output_support_policy import OutputSupportPolicy
from .output_support_adapter import validate_native_arms
from .causal_arm_gate import CausalArmPolicy
from .structured_features import pack_observable
from .eval_diagnostics import array_hash
from .geometry_labels import poses7

CASES=('C_base','A_grip','T_stage_grip')
GRIPPERS=(6,13)
def compose_grip(base,raw,case,active,goal=None):
    if case not in CASES or type(active)is not bool:raise ValueError('Declared probe mode/activation')
    for v in (base,raw):
        if not isinstance(v,np.ndarray) or v.dtype!=np.float32 or v.shape!=(32,14) or not np.isfinite(v).all():
            raise ValueError('Native finitefloat32 currentbase/proposal')
    if case=='T_stage_grip':
        if not isinstance(goal,np.ndarray) or goal.dtype!=np.float32 or goal.shape!=(2,) or not np.isfinite(goal).all() or np.any((goal<0)|(goal>1)):
            raise ValueError('Canonical successfulexpert terminal gripper vector required')
    elif goal is not None:raise ValueError('Privileged reference forbidden in base/frozenNN branches')
    actions=base.copy();delta=np.zeros_like(base)
    if active and case!='C_base':
        if case=='A_grip':delta[:,GRIPPERS]=raw[:,GRIPPERS]
        else:delta[:,GRIPPERS]=np.clip(goal[None]-base[:,GRIPPERS],np.float32(-.2),np.float32(.2))
        nonzero=delta!=0;actions[nonzero]=base[nonzero]+delta[nonzero]
    if np.any(np.abs(delta[:,GRIPPERS])>np.float32(.2)+1e-6):raise ValueError('Original GRIP support exceeded')
    if actions[:,ARM_INDICES].tobytes()!=base[:,ARM_INDICES].tobytes():raise ValueError('Probe arm commands changed')
    return actions,delta

def capture_all(policy,previous,base):
    joint,state=policy.previous
    old_joint,old_state=previous if previous is not None else (np.zeros_like(joint),np.zeros_like(state))
    x=pack_observable(joint[None],state[None],old_joint[None],old_state[None],np.asarray([previous is not None]),policy.projection,policy.robots)
    if set(x)!=FORWARD_KEYS or any(array_hash(v)!=policy.last_diagnostics['structured_input_sha256'][k] for k,v in x.items()):
        raise ValueError('All-query legal packet mismatch')
    policy.last_legal_inputs={**{k:v.copy() for k,v in x.items()},'base':base[None].copy()}

class AllQuerySource(CausalArmPolicy):
    def predict(self,observation,instruction,noise,*,enabled):
        previous=self.previous;actions,delta,base=super().predict(observation,instruction,noise,enabled=enabled)
        capture_all(self,previous,base);return actions,delta,base

class PhaseGripperPolicy(OutputSupportPolicy):
    def __init__(self,base,training_run,case):
        if case not in CASES:raise ValueError('Probe case')
        super().__init__(base,training_run,'A_bounded_clipped',force_zero=True)
        self.condition=case;self.point_query=None;self.teacher_goal=None
    def predict(self,observation,instruction,noise,*,enabled):
        previous=self.previous;q=self.query_count
        _,_,base=super().predict(observation,instruction,noise,enabled=enabled)
        raw=self.last_numeric['raw_delta'];active=q==self.point_query
        actions,delta=compose_grip(base,raw,self.condition,active,self.teacher_goal)
        domain=validate_native_arms(actions,self.joint_limits)
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(probe_active=bool(active),probe_point_query=self.point_query,
            raw_scope='Frozen originalA proposal in everycase',teacher_privileged=self.condition=='T_stage_grip',
            teacher_gripper_goal=None if self.teacher_goal is None else self.teacher_goal.tolist(),
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),
            arm_strength=0.,gripper_strength=float(active and self.condition!='C_base'),robot_domain=domain)
        capture_all(self,previous,base);return actions,delta,base

def functional_state(snapshot,task,definitions,names):
    world=[]
    for name,d in zip(names,definitions):
        matches=[a['pose'] for a in snapshot['actors'] if a['name']==name]
        if len(matches)!=1 or d['functional_local'] is None:raise ValueError('Unambiguous offline functional role')
        world.append(poses7(matches[0])@d['functional_local'])
    world=np.stack(world);diff=world[0,:3,3]-world[1,:3,3]
    geom=bool(np.linalg.norm(diff[:2])<.035 and abs(diff[2])<.015) if task=='place_empty_cup' else bool(np.all(np.abs(diff[:2])<.02))
    return dict(scope='offline_only,currentfunctionalpose,not actor/router input',functional_xyz_m=world[:,:3,3].tolist(),
        goal_xy_cm=float(100*np.linalg.norm(diff[:2])),goal_z_cm=float(100*abs(diff[2])),
        geometry_satisfied=geom,native_success=snapshot['control']['eval_success'],
        command_grippers=[snapshot['control'][s+'_command'][-1] for s in ('left','right')],
        native_count=snapshot['control']['take_action_cnt'],native_budget=snapshot['control']['step_lim'])

