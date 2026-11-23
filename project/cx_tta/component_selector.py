"""Teacher-only component/sequence oracles over immutable action candidates."""
import itertools
import numpy as np
from .candidate_diagnostic import SUPPORTS,CAPS,TOL,material_benefit

STATES=tuple(itertools.product(range(4),repeat=2)) # arm scale index, gripper scale index
RULES={f'{support}_{coupling}_{guard}_{timing}':dict(support=support,coupling=coupling,guard=guard,timing=timing)
    for support,coupling,guard,timing in itertools.product(('raw','capped'),('tied','split'),('strict','tolerant'),('local','sequence'))}
FLOORS={'angle_deg':1.,'position_m':.01,'gripper_mae':.02}


def component_grid(fixed,metrics,target,valid,support):
    """FK depends on arm alpha only; substitute the independently scaled finger."""
    pool=SUPPORTS[support];count=valid.sum(1)[:,None]
    grip_mse=[]
    for index in pool:
        delta=fixed[index][...,[6,13]]-target[...,[6,13]]
        grip_mse.append((delta**2*valid[:,:,None]).sum(1)/count)
    output=[]
    for arm,grip in STATES:
        m={k:v.copy() for k,v in metrics[pool[arm]].items()}
        if arm!=grip:
            for key in ('gripper_mae','gripper_outside_margin'):
                m[key]=metrics[pool[grip]][key].copy()
            change=(grip_mse[grip]-grip_mse[arm])/(7*.2**2)
            for key in ('dense_bc','score'):m[key]=m[key]+change
        m['arm_score']=metrics[pool[arm]]['arm_mse_rad2']*(6/7)/.1**2+metrics[pool[arm]]['rotation_loss']+metrics[pool[arm]]['position_loss']
        m['grip_mse']=grip_mse[grip]
        output.append(m)
    return {k:np.stack([m[k] for m in output]) for k in output[0]}


def feasible(grid,coupling,guard):
    allowed=np.ones_like(grid['score'],bool)
    if coupling=='tied':
        allowed&=np.array([a==g for a,g in STATES])[:,None,None]
    for key,tol in TOL.items():
        threshold=grid[key][0] if guard=='strict' else np.maximum(grid[key][0],FLOORS[key])
        allowed&=grid[key]<=threshold+tol
    for key in ('gripper_outside_margin','joint_limit_violations'):
        allowed&=grid[key]<=grid[key][0]
    if not allowed[0].all():raise ValueError('Zero action must be admissible')
    return allowed


def assemble(fixed,support,selected):
    pool=SUPPORTS[support];out=fixed[0].copy();state=np.asarray(STATES)[selected]
    for side,start in enumerate((0,7)):
        for alpha in range(4):
            mask=state[:,side,0]==alpha;out[mask,:,start:start+6]=fixed[pool[alpha]][mask,:,start:start+6]
            mask=state[:,side,1]==alpha;out[mask,:,start+6]=fixed[pool[alpha]][mask,:,start+6]
    return out


def edge_matrix(fixed,support,target,pair,side):
    pool=SUPPORTS[support];start=7*side;b,t,a=pair['before'],pair['last'],pair['after']
    previous=np.stack([np.r_[fixed[pool[x]][b,t,start:start+6],fixed[pool[y]][b,t,start+6]] for x,y in STATES])
    current=np.stack([np.r_[fixed[pool[x]][a,0,start:start+6],fixed[pool[y]][a,0,start+6]] for x,y in STATES])
    desired=target[a,0,start:start+7]-target[b,t,start:start+7]
    return np.mean(((current[None]-previous[:,None]-desired)/CAPS[start:start+7])**2,-1)


def choose(grid,setting,fixed,target,chains):
    allowed=feasible(grid,setting['coupling'],setting['guard']);unary=np.where(allowed,grid['score'],np.inf)
    choice=np.argmin(unary,axis=0).astype(np.int16);records=[]
    for chain in chains:
        ids=chain['rows'];n=len(ids);edges=chain['pairs']
        for side in (0,1):
            transitions=[edge_matrix(fixed,setting['support'],target,e,side)/len(edges) for e in edges]
            local=choice[ids,side].copy()
            if setting['timing']=='sequence':
                cost=unary[:,ids[0],side]/n;parents=[]
                for j,edge in enumerate(transitions,1):
                    candidates=cost[:,None]+edge;parent=np.argmin(candidates,axis=0);parents.append(parent)
                    cost=unary[:,ids[j],side]/n+candidates[parent,np.arange(16)]
                optimum=int(np.argmin(cost));selected=[optimum]
                for parent in reversed(parents):selected.append(int(parent[selected[-1]]))
                choice[ids,side]=selected[::-1]
            selected=choice[ids,side]
            dense=float(sum(unary[s,i,side] for s,i in zip(selected,ids))/n)
            seam=float(sum(edge[selected[j],selected[j+1]] for j,edge in enumerate(transitions)))
            local_total=float(sum(unary[s,i,side] for s,i in zip(local,ids))/n+
                sum(edge[local[j],local[j+1]] for j,edge in enumerate(transitions)))
            if dense+seam>local_total+1e-8:raise ValueError('Sequence optimum worse than independent choices')
            records.append(dict(group=chain['group'],arm=('left','right')[side],dense=dense,seam=seam,
                total=dense+seam,independent_total=local_total,choices=selected.tolist()))
    return choice,records


def benefit_flags(grid):
    base={k:v[0] for k,v in grid.items()}
    joint=np.stack([material_benefit({k:v[i] for k,v in grid.items()},base) for i in range(16)])
    arm=[];grip=[]
    for index in range(4):
        a=index*4;g=index
        arm.append((grid['arm_score'][0]-grid['arm_score'][a]>np.maximum(1e-8,.05*grid['arm_score'][0]))&
            (grid['angle_deg'][a]<=grid['angle_deg'][0]+1e-6)&(grid['position_m'][a]<=grid['position_m'][0]+1e-9)&
            (grid['joint_limit_violations'][a]<=grid['joint_limit_violations'][0]))
        grip.append((grid['grip_mse'][0]-grid['grip_mse'][g]>np.maximum(1e-8,.05*grid['grip_mse'][0]))&
            (grid['gripper_mae'][g]<=grid['gripper_mae'][0]+1e-8)&
            (grid['gripper_outside_margin'][g]<=grid['gripper_outside_margin'][0]))
    return dict(joint=joint,arm=np.stack(arm),grip=np.stack(grip))
