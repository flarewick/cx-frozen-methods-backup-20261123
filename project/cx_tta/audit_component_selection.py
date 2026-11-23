"""Independent exhaustive chain optimum and native-trajectory metric audit."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .probe_candidates import read,verify,load_candidate
from .recovery_fit_metrics import metrics as original_metrics
from .audit_time_encoding_dense import strict_fit,preservation
from .robot_kinematics import RobotFK
from .audit_pose_capacity import numpy_fk


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();threadpool_limits(2)
    p=read(run/'protocol.json');r=read(run/'result.json');verify(p)
    if r['status']!='complete' or len(r['reports'])!=12 or len(p['rules'])!=16:raise ValueError('Incomplete factorial diagnostic')
    source=read(Path(p['reference_source'])/'result.json')
    with np.load(p['data']) as z:arrays={k:z[k].copy() for k in z.files}
    base=arrays['base'].astype(float);target=arrays['target'].astype(float);valid=arrays['valid'];rows=p['rows']
    bots=[RobotFK(read(Path(p['config'])),side) for side in ('left','right')]
    caps=np.array([.1]*6+[.2]+[.1]*6+[.2]);states=np.array(list(itertools.product(range(4),repeat=2)))
    pools={'raw':[0,1,2,3],'capped':[0,4,5,6]};source_metrics=['angle_deg','position_m','gripper_mae','gripper_outside_margin','joint_limit_violations']
    max_score=0.;max_optimum=0.;max_fk=0.;count=0;exhaustive=0;files=[]
    example=np.array(sorted({next(i for i,row in enumerate(rows) if row['group']==g) for g in {r['group'] for r in rows}}|set(p['parts']['recovery'])))
    for report,model,old in zip(r['reports'],p['models'],source['reports']):
        if not (report['condition']==model['condition']==old['condition'] and report['seed']==model['seed']==old['seed']):raise ValueError('Wrong model pairing')
        file=Path(report['panel']);files.append(file)
        if digest(file)!=report['panel_sha256']:raise ValueError('Changed diagnostic panel')
        with np.load(file) as z:panel={k:z[k].copy() for k in z.files}
        with np.load(old['panel']) as z:original={k:z[k].copy() for k in z.files}
        pred=load_candidate(model,p,arrays).astype(float);delta=pred-base;clip=np.maximum(-caps,np.minimum(caps,delta))
        fixed=[base,base+.25*delta,base+.5*delta,pred,base+.25*clip,base+.5*clip,base+clip]
        for support,pool in pools.items():
            # Direct action construction and native7-channel score, independent of grip-substitution algebra.
            commands=[];grid=[]
            for arm,grip in states:
                command=fixed[pool[arm]].copy();command[...,[6,13]]=fixed[pool[grip]][...,[6,13]];commands.append(command)
                diff=(command-target).reshape(len(base),32,2,7)/caps.reshape(2,7)
                score=(np.mean(diff**2,-1)*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
                score+=original['rotation_loss'][pool[arm]]+original['position_loss'][pool[arm]]
                grid.append(score)
            grid=np.stack(grid);error=float(np.abs(grid-panel[support+'_grid_score']).max());max_score=max(max_score,error)
            if error>1e-8:raise ValueError('Component substitution score mismatch')
            arm_cost=[];finger_cost=[]
            for idx in pool:
                diff=(fixed[idx]-target).reshape(len(base),32,2,7)
                avg=lambda x:(x*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
                arm_cost.append(avg(np.sum(diff[...,:6]**2,-1)/(7*.1**2))+original['rotation_loss'][idx]+original['position_loss'][idx])
                finger_cost.append(avg(diff[...,6]**2))
            joint_flags=[];arm_flags=[];grip_flags=[]
            for state,(aa,gg) in enumerate(states):
                a,g=pool[aa],pool[gg]
                keep=(grid[0]-grid[state]>np.maximum(1e-8,.05*grid[0]))
                for key,idx,tol in [('angle_deg',a,1e-6),('position_m',a,1e-9),('gripper_mae',g,1e-8),('joint_limit_violations',a,0),('gripper_outside_margin',g,0)]:
                    keep&=original[key][idx]<=original[key][0]+tol
                joint_flags.append(keep)
            for i,idx in enumerate(pool):
                arm_flags.append((arm_cost[0]-arm_cost[i]>np.maximum(1e-8,.05*arm_cost[0]))&
                    (original['angle_deg'][idx]<=original['angle_deg'][0]+1e-6)&
                    (original['position_m'][idx]<=original['position_m'][0]+1e-9)&
                    (original['joint_limit_violations'][idx]<=original['joint_limit_violations'][0]))
                grip_flags.append((finger_cost[0]-finger_cost[i]>np.maximum(1e-8,.05*finger_cost[0]))&
                    (original['gripper_mae'][idx]<=original['gripper_mae'][0]+1e-8)&
                    (original['gripper_outside_margin'][idx]<=original['gripper_outside_margin'][0]))
            for kind,expected in [('joint',joint_flags),('arm',arm_flags),('grip',grip_flags)]:
                if not np.array_equal(np.stack(expected),panel[support+'_'+kind+'_benefit']):raise ValueError('Component benefit label mismatch')
            for vi,(name,setting) in enumerate(p['rules'].items()):
                if setting['support']!=support:continue
                selected=panel['choices'][vi];allowed=np.ones_like(grid,bool)
                if setting['coupling']=='tied':allowed&=(states[:,0]==states[:,1])[:,None,None]
                for key,tol in [('angle_deg',1e-6),('position_m',1e-9),('gripper_mae',1e-8),('gripper_outside_margin',0),('joint_limit_violations',0)]:
                    values=np.stack([original[key][pool[g if key.startswith('gripper') else a]] for a,g in states])
                    threshold=original[key][0]
                    if setting['guard']=='tolerant' and key in p['tolerance_floors']:threshold=np.maximum(threshold,p['tolerance_floors'][key])
                    allowed&=values<=threshold+tol
                if not allowed[0].all() or not np.all(allowed[selected,np.arange(len(base))[:,None],np.arange(2)]):raise ValueError('Inadmissible selection')
                unary=np.where(allowed,grid,np.inf);local=np.argmin(unary,axis=0)
                connected={i for chain in p['chains'] for i in chain['rows']};unconnected=[i for i in range(len(base)) if i not in connected]
                selected_unary=unary[selected,np.arange(len(base))[:,None],np.arange(2)]
                if np.max(selected_unary[unconnected]-unary.min(0)[unconnected])>1e-8:raise ValueError('Invented nonconsecutive edge')
                if setting['timing']=='local' and np.max(selected_unary-unary.min(0))>1e-8:raise ValueError('Independent minimization mismatch')
                action=base.copy()
                for state in range(16):
                    for side,start in enumerate((0,7)):
                        mask=selected[:,side]==state;action[mask,:,start:start+7]=commands[state][mask,:,start:start+7]
                if support=='capped' and np.max(np.abs(action-base)-caps)>1e-12:raise ValueError('Capped support violated')
                gripper=action[...,[6,13]];bg=base[...,[6,13]]
                if ((gripper<np.minimum(bg,0)-1e-12)|(gripper>np.maximum(bg,1)+1e-12)).any():raise ValueError('Gripper envelope violated')
                for chain in p['chains']:
                    ids=chain['rows'];n=len(ids);ne=len(chain['pairs'])
                    combinations=np.array(list(itertools.product(range(16),repeat=n)),dtype=int)
                    for side,start in enumerate((0,7)):
                        edge=[]
                        for e in chain['pairs']:
                            before=np.stack([a[e['before'],e['last'],start:start+7] for a in commands]);after=np.stack([a[e['after'],0,start:start+7] for a in commands])
                            desired=target[e['after'],0,start:start+7]-target[e['before'],e['last'],start:start+7]
                            edge.append(np.mean(((after[None]-before[:,None]-desired)/caps[start:start+7])**2,-1)/ne)
                        costs=sum(unary[combinations[:,j],i,side]/n for j,i in enumerate(ids))
                        costs+=sum(e[combinations[:,j],combinations[:,j+1]] for j,e in enumerate(edge))
                        seq=selected[ids,side];actual=sum(unary[s,i,side]/n for s,i in zip(seq,ids))+sum(e[seq[j],seq[j+1]] for j,e in enumerate(edge))
                        record=next(x for x in report['chain_objectives'][name] if x['group']==chain['group'] and x['arm']==('left','right')[side])
                        if abs(actual-record['total'])>1e-8:raise ValueError('Reported path objective mismatch')
                        if setting['timing']=='sequence':
                            error=abs(actual-float(costs.min()));max_optimum=max(max_optimum,error);exhaustive+=1
                            if error>1e-8:raise ValueError('DP is not globally optimal over frozen finite chain')
                ref,geom=original_metrics(action,base,target,valid,rows,bots);lookup={g['group']:g for g in ref['groups']}
                oi=np.array(p['parts']['ordinary']);ore,_=original_metrics(action[oi],base[oi],target[oi],valid[oi],[rows[i] for i in oi],bots)
                for part,summary in report['summaries'][name].items():
                    expected=ore['groups'] if part=='ordinary' else [lookup[g['group']] for g in summary['groups']]
                    for actual,want in zip(summary['groups'],expected):
                        for side in ('left','right'):
                            for key,value in actual['arms'][side].items():
                                if key in ('score','base_score'):continue
                                other=want['arms'][side][key]
                                if other is None:
                                    if value is not None:raise ValueError('Empty metric differs')
                                elif abs(value-other)>1e-8:raise ValueError('Native group metric mismatch '+key)
                        if actual['seam_error_rmse_rad']!=want['seam_error_rmse_rad']:raise ValueError('Selected seam not recomputed')
                    if strict_fit(dict(groups=expected))!=summary['recovery_fit_gate'] or preservation(dict(groups=expected),anchor=True)!=summary['preservation_gate']:
                        raise ValueError('Gate mismatch')
                    for task,values in summary['tasks'].items():
                        for key,value in values.items():
                            values2=[a[key] for g in summary['groups'] if g['task']==task for a in g['arms'].values() if a[key] is not None]
                            if value!=(None if not values2 else float(np.mean(values2))):raise ValueError('Task mean mismatch')
                for side,start in enumerate((0,7)):
                    q=np.stack([action[i,t,start:start+6] for i in example for t in (0,int(valid[i].sum())-1)])
                    vector=numpy_fk(bots[side],q)
                    for j,value in enumerate(q):
                        err=float(np.abs(bots[side].pose(value)-vector[j]).max());max_fk=max(max_fk,err)
                        if err>1e-10:raise ValueError('Scalar FK mismatch')
                count+=1
        print('AUDITED',report['condition'],report['seed'],flush=True)
    verify(p)
    write_json(run/'audit.json',dict(passed=True,models=12,rules_per_model=16,model_rule_checks=count,
        exhaustive_chain_arm_optima=exhaustive,max_dp_bruteforce_error=max_optimum,
        component_score_formula_max_error=max_score,scalar_fk_max_error=max_fk,
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',Path(__file__),*files]},
        scope='Checks implementation and teacher-known finite-grid optima, not causal routing or physical success.'))
    print('PASSED',count,exhaustive,max_optimum,max_score,max_fk,flush=True)

if __name__=='__main__':main()
