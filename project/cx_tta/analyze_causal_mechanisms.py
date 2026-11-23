"""S2s: offline functional geometry, actual FK, first-chunk effects and legal support.
Never changes an actor/router, fits held scenes, imports a simulator or queries VLA.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .paths import ROOT,write_json
from .audit import digest
from .native_history_cache import read,arrays,require,verify,BASE
from .geometry_labels import episode_definition,poses7
from .robot_kinematics import RobotFK
from .audit_pose_capacity import numpy_fk
from .structured_attribution_analysis import _load_trace
from .structured_action_adapter import StructuredActionAdapter,ARM_INDICES
from .structured_action_deploy import read_checkpoint
from .eval_diagnostics import array_hash

PARENT=ROOT/'outputs/reports/attribution/causal_arm_accepted_20261006_082020_945622/result.json'
SIDES=('left','right')
CONDITIONS=('Z_zero','A_causal_arm')

def official_tcp(link,config,side):
    c=config[side+'_embodiment_config'];out=link.copy()
    out[...,:3,:3]=link[...,:3,:3]@np.asarray(c['global_trans_matrix'])@np.asarray(c['delta_matrix'])
    out[...,:3,3]=link[...,:3,3]+out[...,:3,:3]@np.array([c['gripper_bias'],0,0])
    return out

def qmap(snapshot):
    result={}
    for art in snapshot['articulations']:
        offset=0
        for j in art['active_joints']:
            dof=j['dof'];require(type(dof)is int and dof>0,'Active joint DOF')
            if j['name'] in result:raise ValueError('Duplicate joint name across articulations')
            result[j['name']]=(np.asarray(art['qpos'][offset:offset+dof],float),poses7(art['root_pose']))
            offset+=dof
        require(offset==len(art['qpos']),'qpos coverage')
    return result

def snapshot_geometry(snapshot,task,definitions,cfg,robots,actor_names):
    require(snapshot['usable'],'Readable native state required')
    roots=[]
    for name in actor_names:
        matches=[a['pose'] for a in snapshot['actors'] if a['name']==name]
        require(len(matches)==1,'Unambiguous role actor '+name);roots.append(poses7(matches[0]))
    roots=np.stack(roots);functional=np.stack([root@d['functional_local'] for root,d in zip(roots,definitions)])
    contacts=[np.stack([roots[i]@p for p in definitions[i]['contact_local']]) if len(definitions[i]['contact_local']) else np.empty((0,4,4)) for i in (0,1)]
    states=qmap(snapshot);links=[];commands=[];fractions=[];joint_errors=[]
    for side,robot in zip(SIDES,robots):
        values=[states[n] for n in robot.names];require(all(len(q)==1 for q,_ in values),'OneDOF arm joints')
        root=values[0][1];require(all(np.array_equal(root,r) for _,r in values),'Arm shares a native root')
        q=np.asarray([v[0][0] for v in values])
        real=root@np.linalg.inv(robot.root)@robot.pose(q)
        cmd=np.asarray(snapshot['control'][side+'_command'],float)
        links.append(official_tcp(real,cfg,side));commands.append(official_tcp(robot.pose(cmd[:6]),cfg,side))
        joint_errors.append(float(np.max(np.abs(q-cmd[:6]))))
        c=cfg[side+'_embodiment_config'];idx=0 if side=='left' else 1
        gripper=states[c['gripper_name'][idx]['base']][0];require(len(gripper)==1,'Scalar gripper qpos')
        low,high=c['gripper_scale'];fractions.append(float((gripper[0]-low)/(high-low)))
    tcp=np.stack(links);command_tcp=np.stack(commands);diff=functional[0,:3,3]-functional[1,:3,3]
    grip=[snapshot['control'][s+'_command'][-1] for s in SIDES]
    xy=float(np.linalg.norm(diff[:2]));z=float(abs(diff[2]));full=float(np.linalg.norm(diff))
    geom=bool(xy<.035 and z<.015) if task=='place_empty_cup' else bool(np.all(np.abs(diff[:2])<.02))
    opened=all(v>.8 for v in grip)
    d=dict(native_count=snapshot['control']['take_action_cnt'],native_success=snapshot['control']['eval_success'],
        functional_xyz_m=functional[:,:3,3].tolist(),source_to_destination_m=(-diff).tolist(),
        goal_xy_cm=xy*100,goal_z_cm=z*100,goal_full_cm=full*100,official_geometric_predicate=geom,
        command_grippers=grip,measured_base_gripper_joint_fraction=fractions,official_both_command_open=opened,
        cup_full_predicate=bool(geom and opened) if task=='place_empty_cup' else None,
        hammer_contact_unknown=task=='beat_block_hammer',actual_tcp_xyz_m=tcp[:,:3,3].tolist(),
        command_vs_actual_tcp_distance_cm=(100*np.linalg.norm(tcp[:,:3,3]-command_tcp[:,:3,3],axis=1)).tolist(),
        command_vs_actual_arm_joint_max_rad=joint_errors,
        command_vs_actual_tcp_angle_deg=np.degrees(Rotation.from_matrix(tcp[:,:3,:3]@command_tcp[:,:3,:3].swapaxes(-1,-2)).magnitude()).tolist(),
        tcp_to_each_primary_contact_cm=(100*np.linalg.norm(tcp[:,None,:3,3]-contacts[0][None,:,:3,3],axis=-1)).tolist(),
        source_functional_xyz_in_each_actual_tcp_m=np.einsum('hi,hij->hj',functional[0,:3,3]-tcp[:,:3,3],tcp[:,:3,:3]).tolist())
    return d,dict(roots=roots,functional=functional,contacts=contacts,tcp=tcp,command_tcp=command_tcp)

def planned_metrics(t,i,cfg,robots,g):
    b=t['arrays']['base'][i];raw=t['arrays']['raw_delta'][i];p=b.copy();p[:,ARM_INDICES]+=raw[:,ARM_INDICES]
    output=[];maxparity=0.
    for side,start,robot in zip(SIDES,(0,7),robots):
        bl=numpy_fk(robot,b[:,start:start+6].astype(float));pl=numpy_fk(robot,p[:,start:start+6].astype(float))
        for actions,vectorized in ((b,bl),(p,pl)):
            scalar=np.stack([robot.pose(q) for q in actions[:,start:start+6].astype(float)])
            maxparity=max(maxparity,float(np.abs(scalar-vectorized).max()))
        bt,pt=official_tcp(bl,cfg,side),official_tcp(pl,cfg,side);delta=pt[:,:3,3]-bt[:,:3,3]
        angle=np.degrees(Rotation.from_matrix(pt[:,:3,:3]@bt[:,:3,:3].swapaxes(-1,-2)).magnitude())
        contact=g['contacts'][0][:,:3,3];base_d=np.linalg.norm(bt[:,None,:3,3]-contact[None],axis=-1)
        proposed_d=np.linalg.norm(pt[:,None,:3,3]-contact[None],axis=-1)
        goal=g['functional'][1,:3,3]-g['functional'][0,:3,3];n=np.linalg.norm(goal);norm=np.linalg.norm(delta,axis=1)
        cos=np.divide(delta@goal,n*norm,out=np.zeros(32),where=n*norm>1e-10)
        row=dict(side=side,joint_delta_RMS_rad=float(np.sqrt(np.mean(raw[:,start:start+6]**2))),
            joint_delta_max_rad=float(np.abs(raw[:,start:start+6]).max()),
            cap_near95percent_coordinate_fraction=float(np.mean(np.abs(raw[:,start:start+6])>=.095)),
            planned_tcp_delta_RMS_cm=float(100*np.sqrt(np.mean(delta**2))),
            planned_tcp_rotation_mean_deg=float(angle.mean()),planned_tcp_rotation_max_deg=float(angle.max()),
            static_each_contact_distance_change_cm_mean=(100*(proposed_d-base_d).mean(axis=0)).tolist(),
            static_each_contact_distance_change_cm_h0=(100*(proposed_d-base_d)[0]).tolist(),
            static_each_contact_distance_change_cm_h31=(100*(proposed_d-base_d)[-1]).tolist(),
            frozen_source_destination_direction_cosine_mean=float(cos[norm>1e-10].mean()) if np.any(norm>1e-10) else None,
            placement_alignment_scope='Offline static function-point vector; actual carrying/contact unknown. No contactpoint selection,teacher wrist target ordirection-correctness label.')
        output.append(row)
    require(maxparity<1e-10,'Scalar/vectorized independent FK parity')
    return output,maxparity

def support(train_x,train_y,rows,queries):
    blocks=(('current',('visual','text','state','ee_pose')),('past',('past_visual','past_text','past_state','past_ee_pose')),('base',()))
    weights=np.asarray(train_y['row_weights'],float);weights/=weights.sum();dist=np.zeros((len(queries),len(rows)))
    stats=[]
    for name,keys in blocks:
        ref=train_y['base'].reshape(len(rows),-1).astype(float) if not keys else np.concatenate([train_x[k].reshape(len(rows),-1) for k in keys],1).astype(float)
        values=np.stack([q['packet']['base'][0].reshape(-1) if not keys else np.concatenate([q['packet'][k][0].reshape(-1) for k in keys]) for q in queries]).astype(float)
        mean=weights@ref;scale=float(np.sqrt(weights@np.mean((ref-mean)**2,axis=1)));require(scale>0,'Nonzero training-only block RMS')
        block=np.maximum(((values**2).sum(1)[:,None]+(ref**2).sum(1)[None]-2*values@ref.T)/(ref.shape[1]*scale**2),0)
        dist+=block/len(blocks);stats.append(dict(block=name,dimensions=ref.shape[1],train_only_RMS=scale))
    counts=[]
    h=train_x['has_history'].reshape(-1)>0
    for task in sorted({r['task'] for r in rows}):
        for kind in ('expert','anchor','recovery'):
            ids=[i for i,r in enumerate(rows) if r['task']==task and r['kind']==kind]
            counts.append(dict(task=task,kind=kind,rows=len(ids),history_rows=int(h[ids].sum())))
    out=[]
    for i,q in enumerate(queries):
        take=np.argsort(dist[i])[:3];mask=h==bool(q['packet']['has_history'][0,0]);matching=np.flatnonzero(mask);nearest=matching[np.argmin(dist[i,matching])] if len(matching) else None
        out.append(dict(task=q['task'],block=q['block'],seed=q['seed'],query=q['query'],history=bool(q['packet']['has_history'][0,0]),
            top3=[dict(kind=rows[n]['kind'],task=rows[n]['task'],group=rows[n]['group'],row=int(n),distance=float(np.sqrt(dist[i,n]))) for n in take],
            history_matching_rows=len(matching),history_matching_nearest=None if nearest is None else dict(row=int(nearest),task=rows[nearest]['task'],kind=rows[nearest]['kind'],distance=float(np.sqrt(dist[i,nearest]))),
            scope='Descriptive fixed training-only block-RMS distance,not calibrated benefit/risk confidence orroutingthreshold'))
    return dict(normalization=stats,training_history_coverage=counts,first_fork_queries=out,no_query_labels_fit=True)

def run(report):
    parent=read(PARENT);require(parent['status']=='passed','Accepted S2r input panel');torch.set_num_threads(2)
    protocol=read(report/'protocol.json');verify(protocol['source_sha256'])
    registry=read(Path(parent['registry'])/'records.json');blocks=[];contexts=[];timelines=[];fork_packets=[];sp_packets=[];sp_truth=[]
    source_sha=dict(protocol['source_sha256']);parities=[];predicate_checks=0;input_fk_errors=[];qpos_maps=0
    train_x=arrays(BASE/'inputs.npz');train_y=arrays(BASE/'labels.npz');train_rows=read(BASE/'protocol.json')['rows']
    for block in parent['blocks']:
        path=Path(block['run']);m=read(path/'manifest.json');task=m['task'];cfg=read(path/'resolved_config.json');robots=[RobotFK(cfg,s) for s in SIDES]
        for robot in robots:source_sha[str(robot.urdf)]=digest(robot.urdf)
        source_sha[str(path/'resolved_config.json')]=digest(path/'resolved_config.json')
        definitions=None;eps=read(path/'episodes.json');traces={}
        for ep in eps:
            t=_load_trace(path,ep);traces[ep['seed'],ep['condition']]=t
            record=next(r for r in registry if r['task']==task and r['seed']==ep['seed'])
            defs,deps=episode_definition(path/'robotwin',task,record['grounded_objects'])
            names=[record['grounded_objects']['{A}'].split('/')[0],'019_coaster' if task=='place_empty_cup' else 'box']
            for f in deps:source_sha[str(f)]=digest(f)
            scalar=[]
            for i,row in enumerate(t['rows']):
                before,g=snapshot_geometry(row['native_before'],task,defs,cfg,robots,names)
                after,_=snapshot_geometry(row['native_after'],task,defs,cfg,robots,names);qpos_maps+=4
                scalar.append(dict(query=i,active=row['window_active'],executed=row['executed'],before=before,after=after))
            final=scalar[-1]['after']
            if task=='place_empty_cup':
                require(final['cup_full_predicate']==ep['success'],'Exact official cup terminal reconstruction');predicate_checks+=1
            elif ep['success']:
                require(final['official_geometric_predicate'],'Hammer success necessary geometric predicate');predicate_checks+=1
            timelines.append(dict(task=task,block=m['block_id'],seed=ep['seed'],condition=ep['condition'],success=ep['success'],queries=scalar))
            cap=arrays(path/'legal_inputs'/(ep['trace_stem']+'.npz'));ids=cap.pop('query_indices')
            for j,i in enumerate(ids):
                state=t['arrays']['state'][i].astype(float)
                ee=np.stack([numpy_fk(robot,state[start:start+6][None])[0] for robot,start in zip(robots,(0,7))])
                expected=np.concatenate([ee[:,:3,3]/.1,ee[:,:3,:2].reshape(2,6)],axis=1).astype(np.float32)
                input_fk_errors.append(float(np.abs(expected-cap['ee_pose'][j,0]).max()))
                truth,gg=snapshot_geometry(t['rows'][i]['native_before'],task,defs,cfg,robots,names)
                sp_packets.append({k:v[j] for k,v in cap.items()})
                sp_truth.append(dict(task=task,block=m['block_id'],seed=ep['seed'],condition=ep['condition'],query=int(i),
                    primary_functional_xyz=gg['functional'][0,:3,3],destination_functional_xyz=gg['functional'][1,:3,3]))
        for seed in m['seeds']:
            z,a=[traces[seed,c] for c in CONDITIONS];i=a['rows'][-1]['gate_decision']['window'][0];w=a['rows'][-1]['gate_decision']['window']
            require(z['rows'][i]['native_before_sha256']==a['rows'][i]['native_before_sha256'],'Same readable first intervention state')
            require(np.array_equal(z['arrays']['base'][i],a['arrays']['base'][i]) and np.array_equal(z['arrays']['raw_delta'][i],a['arrays']['raw_delta'][i]),'Same actual first candidate/base')
            reg=next(r for r in registry if r['task']==task and r['seed']==seed);defs,deps=episode_definition(path/'robotwin',task,reg['grounded_objects'])
            names=[reg['grounded_objects']['{A}'].split('/')[0],'019_coaster' if task=='place_empty_cup' else 'box']
            before,g=snapshot_geometry(a['rows'][i]['native_before'],task,defs,cfg,robots,names)
            zm,_=snapshot_geometry(z['rows'][i]['native_after'],task,defs,cfg,robots,names);am,_=snapshot_geometry(a['rows'][i]['native_after'],task,defs,cfg,robots,names)
            require(zm['native_count']==am['native_count'],'Same firstchunk execution horizon')
            endz,_=snapshot_geometry(z['rows'][min(w[1]-1,len(z['rows'])-1)]['native_after'],task,defs,cfg,robots,names)
            enda,_=snapshot_geometry(a['rows'][w[1]-1]['native_after'],task,defs,cfg,robots,names)
            planned,parity=planned_metrics(a,i,cfg,robots,g);parities.append(parity)
            packet=arrays(path/'legal_inputs'/(a['episode']['trace_stem']+'.npz'));ix=np.flatnonzero(packet.pop('query_indices')==i);require(len(ix)==1,'Actual firstfork legal capture')
            packet={k:v[ix[0]] for k,v in packet.items()}
            fork_packets.append(dict(task=task,block=m['block_id'],seed=seed,query=i,packet=packet))
            contexts.append(dict(task=task,block=m['block_id'],seed=seed,query=i,
                zero_success=z['episode']['success'],candidate_success=a['episode']['success'],
                outcome='tie' if z['episode']['success']==a['episode']['success'] else 'rescued' if a['episode']['success'] else 'harmed',
                event_decrease=a['rows'][a['rows'][i]['gate_decision']['first_event_query']]['gate_decision']['closing_decrease'],
                shared_before=before,planned_per_arm=planned,first_chunk_zero=zm,first_chunk_candidate=am,
                first_chunk_goal_xy_change_vs_zero_cm=am['goal_xy_cm']-zm['goal_xy_cm'],
                window_end_zero=endz,window_end_candidate=enda,same_window_endpoint_action_count=endz['native_count']==enda['native_count'],
                actual_first_chunk_per_arm_tcp_displacement_difference_cm=(100*(np.asarray(am['actual_tcp_xyz_m'])-zm['actual_tcp_xyz_m'])).tolist(),
                first_chunk_scope='Strict same-start32command intervention with same base; no claim that frozen contact/placement projection is a teacher target. Windowend can have unequal counts if zero already succeeded.'))
    require(max(input_fk_errors)<2e-5,'Command-FK legal capture parity')
    payload,_=read_checkpoint(BASE/'SP_seed0.pt','SP')
    model=StructuredActionAdapter(32,payload['metadata']['geometry_mean']).eval().requires_grad_(False);model.load_state_dict(payload['model'])
    x={k:torch.from_numpy(np.concatenate([v[k] for v in sp_packets],axis=0)) for k in train_x}
    base=torch.from_numpy(np.concatenate([v['base'] for v in sp_packets],axis=0))
    with torch.inference_mode():pred=model(x,base,enabled=False)['geometry'].numpy()
    readout=[]
    for truth,value in zip(sp_truth,pred):
        p=value[6:9]*.1;d=value[9:12]*.1
        readout.append(dict(**{k:v for k,v in truth.items() if not k.endswith('_xyz')},
            primary_functional_error_cm=float(100*np.linalg.norm(p-truth['primary_functional_xyz'])),
            destination_functional_error_cm=float(100*np.linalg.norm(d-truth['destination_functional_xyz'])),
            relative_functional_error_cm=float(100*np.linalg.norm((d-p)-(truth['destination_functional_xyz']-truth['primary_functional_xyz'])))))
    summary=[]
    for task in sorted({r['task'] for r in readout}):
        rs=[r for r in readout if r['task']==task];cs=[r for r in rs if r['condition']=='A_causal_arm' and r['query']==next(c['query'] for c in contexts if c['task']==task and c['block']==r['block'] and c['seed']==r['seed'])]
        summary.append(dict(task=task,observations=len(rs),firstfork_observations=len(cs),
            all_primary_functional_mean_error_cm=float(np.mean([r['primary_functional_error_cm'] for r in rs])),
            firstfork_primary_functional_mean_error_cm=float(np.mean([r['primary_functional_error_cm'] for r in cs])),
            firstfork_destination_mean_error_cm=float(np.mean([r['destination_functional_error_cm'] for r in cs])),
            firstfork_relative_mean_error_cm=float(np.mean([r['relative_functional_error_cm'] for r in cs]))))
    source_sha[str(BASE/'SP_seed0.pt')]=digest(BASE/'SP_seed0.pt');source_sha[str(Path(__file__))]=digest(Path(__file__))
    result=dict(status='passed',GPU_started=False,policy_queries=0,training_updates=0,source_review=str(PARENT),
        complete_rollouts=24,distinct_scenes=6,firstfork_contexts=len(contexts),real_named_joint_FK_arms_reconstructed=qpos_maps,
        official_terminal_predicate_checks=predicate_checks,command_FK_capture_checks=len(input_fk_errors),
        command_FK_capture_max_error=max(input_fk_errors),scalar_vectorized_planned_FK_max_error=max(parities),
        contexts=contexts,feature_support=support(train_x,train_y,train_rows,fork_packets),
        frozen_SP_readout=dict(scope='Existing XYZ-supervised SP diagnostic only,never deployed in thispanel; no orientation/confidence calibration orquerylabeltraining',summary=summary,observations=readout),
        source_sha256=source_sha,limitations=['GT geometry offline only,no learned router orTTA.',
            'RealjointFK from named qpos andnative root,not directly savedmeasuredlinkposes; collisions/compliance unknown.',
            'Functional point andcommandopening recovercup terminalpredicate; hammer contacts not saved so success cannot be fully reconstructed.',
            'No stable holding certificate from command,fingerjointfraction orroot motion.',
            'Static planned projection does not imply feasible path,grasp,correctorientation orwhole-task benefit.',
            'Only12paired scene-block contexts across6scene bundles; no held-data classifier orthresholdfitting.',
            'Known expert-history augmentation negative/partial result retained,not repeated orclaimedonlycause.'])
    verify(protocol['source_sha256'])
    write_json(report/'timelines_offline_only.json',dict(scope='Offline functional/nativeFK snapshots at query boundaries',episodes=timelines))
    write_json(report/'result.json',result)
    print('MECHANISM_AUDIT_PASS',dict(contexts=len(contexts),predicates=predicate_checks,commandFK=max(input_fk_errors),plannedFK=max(parities),SP_summary=summary),flush=True)
    for c in contexts:print('FIRST_CHUNK',c['task'],c['block'],c['seed'],c['outcome'],'xy_change_cm',c['first_chunk_goal_xy_change_vs_zero_cm'],'contactchanges',[p['static_each_contact_distance_change_cm_mean'] for p in c['planned_per_arm']],flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    try:run(args.report.resolve())
    except BaseException:
        import traceback
        f=args.report/'failure.json'
        if not f.exists():write_json(f,dict(traceback=traceback.format_exc(),GPU_started=False))
        raise

if __name__=='__main__':main()
