"""CPU parent: independently reconstruct each arrived local feedback window."""
from pathlib import Path
import numpy as np
from .paths import read,digest
from .contract import load_packet
from native_delayed_query_v1.primitives import old,configuration,functions,T as TTA
from native_delayed_query_v1.predictor import FrozenPredictor
from types import SimpleNamespace

def verify_live(folder,branch,item,*,teacher=True,cpu_only_fixture=False):
    folder=Path(folder);value=read(folder/(branch+'_live_feedback.json'));checks=0
    if cpu_only_fixture:
        assert item['case_id'].startswith('CPU_fixture_')and value['CPU_fixture_only']and not value['real_SAM2_model']
    else:assert value['real_SAM2_model']and not value['CPU_fixture_only']
    assert not value['old_later_masks_imported']and value['online_updates']==0
    source=item['earliest_seed']
    history_folder=folder
    if folder.name=='sequential_reference':
        assert read(folder.parent/'sequential_reference_checks.json')['passed']
        history_folder=folder.parent
    history=load_packet(history_folder/'observed_history.npz')
    if not source['available']:
        assert not value['rows']and all(not q['rows']for q in value['queries'])
        assert value['owner']['actual_tracking_calls']==0 and not(folder/(branch+'_live_masks.npz')).exists()
        return dict(passed=True,checks=2,known=False,rows=0,actual_tracking_calls=0)
    masks=load_packet(folder/(branch+'_live_masks.npz'));mr=value['mask_readouts'];nq=len(history['queries'])
    assert len(mr)==len(masks['masks'])==value['owner']['actual_tracking_calls'];checks+=1
    if teacher:
        reference=read(item['teacher_records'][branch if branch!='Z_sham'else'Z_reference'])
        oldmask=load_packet(reference['masks_file']);n=len(oldmask['masks'])
        assert len(mr)>=n and np.array_equal(masks['global_ticks'][:n],oldmask['global_ticks']);checks+=1
        assert np.array_equal(masks['masks'][:n],oldmask['masks'])and np.array_equal(masks['presence'][:n],oldmask['presence']);checks+=1
        assert [r['raw_logits_sha256']for r in mr[:n]]==reference['raw_logits_sha256'];checks+=1
        assert [r['RGB_sha256']for r in mr[:n]]==reference['RGB_sha256'];checks+=1
    cfg,camera,robots=configuration();seed=load_packet(source['mask_file'])['masks']
    proto,known,_=old.surface.prototype(history['cam0'][0],seed)
    parts=[];images=[history['cam0'][-1]];ticks=[int(masks['global_ticks'][nq-1])];elapsed=[0.]
    command=history['states'][-1].astype(float);origin=ticks[0];total_s=0.
    for name in('origin','following','delayed'):
        if not(folder/(branch+'_'+name+'.npz')).exists():continue
        packet=load_packet(folder/(branch+'_'+name+'.npz'));trace=load_packet(folder/(branch+'_'+name+'_actuation.npz'))
        rgb=load_packet(folder/(branch+'_'+name+'_rgb.npz'));meta=read(folder/(branch+'_'+name+'.json'))
        assert np.array_equal(rgb['initial_0'],images[-1]);checks+=1
        parts.append(dict(kind='origin'if name=='origin'else'following',name=name,start=meta['absolute_start_tick'],
            end=meta['absolute_start_tick']+len(packet['dt_s']),initial=command.copy(),trace=trace,
            query=nq-1+len(parts)))
        images.extend(rgb['samples_0']);ticks.extend((meta['absolute_start_tick']+packet['ticks']).tolist())
        elapsed.extend((total_s+packet['elapsed_s']).tolist());total_s+=float(packet['dt_s'].sum())
        command[old.ARM]=trace['arm_position_target'][-1];command[old.GRIP]=trace['gripper_logical_command'][-1]
    images=np.stack(images);ticks=np.asarray(ticks,np.int64);elapsed=np.asarray(elapsed,float)
    raw=masks['masks'][nq-1:];presence=masks['presence'][nq-1:]
    assert len(images)==len(raw)and np.array_equal(ticks,masks['global_ticks'][nq-1:]);checks+=1
    for image,row in zip(images,mr[nq-1:]):
        import hashlib
        assert hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()==row['RGB_sha256'];checks+=1
    refined=[];valid=[]
    for image,mask,pres in zip(images,raw,presence):
        m,v,_=old.surface.refine(image,mask,pres,proto,known,item['instruction']);refined.append(m);valid.append(v)
    refined=np.stack(refined);valid=np.stack(valid)
    partner=functions(TTA/'native_origin_response_v1/features.py',('partner_features',),dict(np=np,d=SimpleNamespace(old=old))).partner_features
    built=[];past={};missing=0
    for left in range(len(images)-1):
        right=left+1;start,end=int(ticks[left]),int(ticks[right]);dt=float(elapsed[right]-elapsed[left])
        if not 25<=end-start<=125:missing+=1;continue
        segments=[p for p in parts if p['start']<=start<end<=p['end']];assert len(segments)==1
        seg=segments[0];trace=seg['trace'];a=start-seg['start'];b=end-seg['start'];q=seg['initial'].copy()
        if a:q[old.ARM]=trace['arm_position_target'][a-1];q[old.GRIP]=trace['gripper_logical_command'][a-1]
        window={k:trace[k][a:b]for k in('arm_position_target','arm_velocity_target','gripper_logical_command','dt_s')}
        assert abs(float(window['dt_s'].sum())-dt)<1e-9;checks+=1
        p,good,colors=old.flow.corners(images[left],refined[left],valid[left])
        endpoint,alive,_=old.flow.track(images,refined,valid,left,right,p,good,colors)
        supported=[old.supported(p,good,k)and old.supported(p,alive,k)and old.supported(endpoint,alive,k)for k in range(2)]
        targets=[np.median(endpoint[k,alive[k]]-p[k,alive[k]],axis=0)/320 if supported[k]else None for k in range(2)]
        u=None
        for obj in range(2):
            if not supported[obj]:missing+=1;continue
            x=old.context(obj,p,good,colors,q,robots,cfg,camera,np.zeros(10));x=np.r_[x,partner(p,good,colors,obj)]
            previous=past.get((obj,start))
            if previous is not None:
                px,pu,py,pd=previous
                x[34:44]=np.r_[py.astype(float)/pd,pu[12:16]/pd,(pu[36:38]*.1-px[[6,13]]*np.pi)/pd,pd,1.]
            if u is None:u=old.action(q,window,robots,cfg,camera)
            relative=targets[obj]-targets[1-obj]if supported[1-obj]else np.full(2,np.nan)
            row=dict(context=x,command=u,target=targets[obj],start_points=p[obj],end_points=endpoint[obj],point_alive=alive[obj],
                relative_target=relative,relative_available=np.asarray(supported[1-obj],bool),object_index=obj,start_tick=start,end_tick=end,
                dt_s=dt,segment=seg['kind'],initial_points=int(good[obj].sum()),matched_points=int(alive[obj].sum()),command_query=seg['query'])
            built.append(row);past[(obj,end)]=(x,u,targets[obj],dt)
    assert len(built)==len(value['rows']);checks+=1
    if built:
        actual=load_packet(folder/(branch+'_live_local.npz'))
        for i,row in enumerate(built):
            for key in actual:assert np.array_equal(actual[key][i],row[key],equal_nan=True),(branch,i,key);checks+=1
            assert value['rows'][i]=={k:row[k]for k in value['rows'][i]};checks+=1
    assert len(value['missing'])==missing;checks+=1
    receipt=read(folder/(branch+'_origin_receipt.json'));closed=receipt['completion']
    phase=None if not closed else closed[0].get('requested95_tick')
    phase_s=None if phase is None else float(parts[0]['trace']['dt_s'][:phase].sum())
    consumed=False;predictor=FrozenPredictor();packets=0
    for query in value['queries']:
        eligible=[]
        if receipt['feedback_ready']and not consumed:
            for i,row in enumerate(built):
                st=float(elapsed[np.flatnonzero(ticks==row['start_tick'])[0]])
                en=float(elapsed[np.flatnonzero(ticks==row['end_tick'])[0]])
                if st>=phase_s and en-phase_s>=.25 and row['end_tick']<=query['tick']:eligible.append(i)
        assert query['rows']==eligible and query['optimizer_updates']==0;checks+=1
        assert np.array_equal(np.asarray(query['prediction']).reshape(-1,2),predictor.predict([built[i]for i in eligible]));checks+=1
        if eligible:consumed=True;packets+=1
    assert value['queue']['released_feedback_packets']==packets<=1;checks+=1
    return dict(passed=True,checks=checks,known=True,rows=len(built),missing=missing,
                actual_tracking_calls=len(mr),shadow_feedback_packets=packets,
                CPU_state_storage_old_raw_prefix_bitwise=teacher)
