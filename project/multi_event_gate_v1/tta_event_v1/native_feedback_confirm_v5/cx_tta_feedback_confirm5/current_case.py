"""True current-RGB native discovery, explicit sampler fork and following base."""
from pathlib import Path
import copy
import numpy as np
from .paths import read,write,digest
from .throughput import replay_prefix,discard_observation
from .prefix_time import Counter
from .capture import capture_chunk,export_chunk
from .history import save_history,save_query
from .query import compose
from native_nextchunk_v1.query_current import query_current
from native_delayed_query_v1.sampler import DelayedQueryFork as CurrentQueryFork
from native_delayed_query_v1.boundary import native_continue
from .live_feedback import LiveFeedback
from .initial_binding import verify_initial
from native_nextchunk_v1.sampler_fork import full_hash
from cx_tta_clock.runtime import rng_components

def prefix_execute(task,base,check_cancel):
    returned=0;censored=False
    with Counter(float(task.scene.get_timestep())) as counter:
        for command in base:
            check_cancel();before=task.take_action_cnt;task.take_action(command);discard_observation(task)
            if task.take_action_cnt!=before+1:censored=True;break
            returned+=1
    return returned,counter.ticks,censored

def current_at(observed,tick):
    return dict(observed,observation_tick=int(tick))

def row_for_query(proposal,current,q,physics):
    return dict(query=q,rgb=tuple(im.copy() for im in current['rgb']),
        images=copy.deepcopy(current['images']),command_state=current['command_state'].copy(),
        base=proposal['base'].copy(),raw_delta=proposal['raw_delta'].copy(),
        legal={k:v.copy() for k,v in proposal['legal'].items()},
        physics_sha256=physics,actor_images=proposal['actor_images'],
        actor_observation_tick=proposal['actor_observation_tick'],
        actor_image_source=proposal['actor_image_source'])

def reference_equal(folder,other,branch,*,legacy_prefix=False):
    """All saved observable/dense/readout/RNG quantities; ignore only wall/counters."""
    from .contract import load_packet,same
    checks=0
    for segment in (('origin','following')if legacy_prefix else('origin','following','delayed')):
        name=branch+'_'+segment
        left,right=folder/(name+'.npz'),other/(name+'.npz')
        assert left.exists()==right.exists()
        if not left.exists():
            for suffix in ('_unavailable.json',):
                a,b=folder/(name+suffix),other/(name+suffix)
                assert a.exists()==b.exists()
                if a.exists():assert read(a)==read(b)
            continue
        for suffix in ('.npz','_actuation.npz','_rgb.npz'):
            assert same(load_packet(folder/(name+suffix)),load_packet(other/(name+suffix))),(branch,segment,suffix);checks+=1
        a,b=read(folder/(name+'.json')),read(other/(name+'.json'))
        for key in ('absolute_start_tick','planned_chunk_sha256','initial','initial_images','initial_tokens','samples','prefix_physics_sha256','prefix_rng_sha256','prefix_rng_components'):
            assert a[key]==b[key],(branch,segment,key);checks+=1
        assert read(folder/(name+'_receipt.json'))==read(other/(name+'_receipt.json'));checks+=1
    a,b=read(folder/(branch+'_continuation.json')),read(other/(branch+'_continuation.json'))
    if legacy_prefix:
        for key,val in b.items():assert a[key]==val,(branch,key)
    else:assert a==b
    checks+=1
    for suffix in ('.npz','.json'):
        name=branch+'_following_query'+suffix
        a,b=folder/name,other/name
        assert a.exists()==b.exists()
        if a.exists():
            if suffix=='.npz':assert same(load_packet(a),load_packet(b))
            else:
                x,y=read(a),read(b)
                for key in ('query','actor_images','actor_observation_tick','actor_image_source','rng_before','rng_after','base_sha256'):
                    assert x[key]==y[key],(branch,key)
            checks+=1
    if not legacy_prefix:
        for suffix in ('.npz','.json'):
            a,b=folder/(branch+'_delayed_query'+suffix),other/(branch+'_delayed_query'+suffix)
            assert a.exists()==b.exists()
            if a.exists():
                if suffix=='.npz':assert same(load_packet(a),load_packet(b))
                else:
                    x,y=read(a),read(b)
                    for key in ('query','actor_images','actor_observation_tick','actor_image_source','rng_before','rng_after','base_sha256'):assert x[key]==y[key],(branch,key)
                checks+=1
    return checks

def run_current(task,c,cfg,item,spec,index,folder,lane,base,head,robots,projection,observer,routed_seed,note,*,mask_model,seed_masks,feedback_geometry,predictor,cache):
    from cx_tta import recovery_contract as recovery
    from cx_tta.multi_event import EventDetector,EventConfig
    from cx_tta.empirical_onset_gate import candidate_packet
    detector=EventDetector(EventConfig(**spec['event_config']))
    source=CurrentQueryFork(c.rng,lane,base._model,device_uuid=spec['gpu_uuid_runtime'],
        model_sha256=spec['model_identity'])
    prefix=[];previous=None;found=None;rows=[];queries=[];ticks=[];chunks=[];physical_tick=0;reason=None
    discovery=observer();initial_anchor=None
    try:
        current=replay_prefix(task,item,[],cfg,discovery.read,routed_seed)
        verify_initial(current,item)
        initial_anchor=copy.deepcopy(discovery.anchor)
        maximum=item.get('coverage_start_query',0)+4
        for q in range(maximum):
            if not native_continue(task):reason='native_episode_terminated_before_event';break
            note('current_discovery',lane=lane.name,case=item['case_id'],query=q)
            observed=current_at(current,physical_tick)
            op=lambda:query_current(base,head,robots,projection,observed,item['instruction'],previous,decision_tick=physical_tick)
            proposal=c.call(lane,source.draw,op,query=q,tick=physical_tick)
            event,evidence=detector.step(q,candidate_packet(proposal['legal'],proposal['raw_delta']))
            rows.append(row_for_query(proposal,observed,q,discovery.physical()))
            ticks.append(physical_tick);queries.append(dict(query=q,event=event,rng_before=source.notes[-1]['rng_before'],rng_after=source.notes[-1]['rng_after']))
            previous=proposal['previous']
            if event is not None and q>=item.get('coverage_start_query',0):
                found=proposal;initial=current;physics=discovery.physical();break
            if q==maximum-1:reason='no_event_within_fixed_budget';break
            start=physical_tick
            returned,committed,censored=prefix_execute(task,proposal['base'],c.check_cancel)
            physical_tick+=committed
            c.call(lane,source.finish_prefix,returned,end_tick=physical_tick)
            chunks.append(dict(query=q,start_tick=start,end_tick=physical_tick,censored=censored))
            if censored:reason='native_prefix_censored';break
            prefix.append(proposal['base'].copy());current=discovery.read()
    finally:task.close_env(clear_cache=False)
    if not rows:
        write(folder/'unavailable.json',dict(reason=reason,old_developer_retained=True));return
    save_history(folder,rows)
    write(folder/'prefix_clock.json',dict(query_observation_ticks=ticks,executed_chunks=chunks,dt_s=float(np.float32(.004)),
        physical_interval_duration_available=True,native_returned_counts_and_actual_slow_steps=True))
    write(folder/'discovery.json',dict(queries=queries,sampler_calls=len(rows),prefix_chunks=len(prefix),
        actual_actor_images_current=True,sampler_witnesses=source.notes,no_expert_screening=True,uses_outcome_labels=False))
    if reason:
        write(folder/'unavailable.json',dict(reason=reason,old_developer_retained=True));return
    event=queries[-1]['event']
    if len(event['kinds'])!=1:
        write(folder/'unavailable.json',dict(reason='unsupported_multiple_event_kinds',event=event));return
    kind=event['kinds'][0]
    from stage_feedback_v3.completion import CompletionQueue
    try:
        check=CompletionQueue();check.register('event','origin',kind,initial['command_state'][[6,13]],found['base'],decision_tick=0,frame_id='legal_current_RGB')
    except ValueError:
        write(folder/'unavailable.json',dict(reason='unsupported_requested_excursion',kind=kind));return
    with (folder/'proposal.npz').open('xb') as f:np.savez_compressed(f,base=found['base'],raw_delta=found['raw_delta'],prefix=np.stack(prefix) if prefix else np.zeros((0,32,14),np.float32))
    with (folder/'initial_current.npz').open('xb') as f:np.savez_compressed(f,**{k:initial[k] for k in ('joint','visual','command_state')})
    save_query(folder,'selected_query',found,dict(query=len(rows)-1,**{k:source.notes[-1][k] for k in ('rng_before','rng_after','base_sha256')}))
    branch_reports={}
    def run_branch(branch,target,*,sequential=False):
        obs=observer();clock_start=ticks[-1]
        if sequential:
            cur=replay_prefix(task,item,[],cfg,obs.read,routed_seed);prev=None;at=0
            for q,record in enumerate(rows):
                assert native_continue(task)
                now=current_at(cur,at)
                assert obs.physical()==record['physics_sha256']
                assert now['images']==record['actor_images'] and np.array_equal(now['command_state'],record['command_state'])
                proposal=c.call(lane,query_current,base,head,robots,projection,now,item['instruction'],prev,decision_tick=at)
                for key in ('base','raw_delta'):assert np.array_equal(proposal[key],record[key]),('reference_query',q,key)
                for key in record['legal']:assert np.array_equal(proposal['legal'][key],record['legal'][key]),('reference_legal',q,key)
                prev=proposal['previous']
                if q==len(rows)-1:
                    initial_obs=cur;selected=proposal;break
                returned,count,censor=prefix_execute(task,proposal['base'],c.check_cancel)
                assert returned==32 and not censor
                at+=count;assert at==ticks[q+1];cur=obs.read()
            fork=None
        else:
            obs.anchor=copy.deepcopy(initial_anchor)
            initial_obs=replay_prefix(task,item,prefix,cfg,obs.read,routed_seed)
            assert obs.physical()==physics
            for key in ('joint','visual','command_state'):assert np.array_equal(initial_obs[key],initial[key]),('fork_current',key)
            selected=found;prev=found['previous']
            fork=c.call(lane,source.branch,branch,c.rng,lane,device_uuid=spec['gpu_uuid_runtime'],model_sha256=spec['model_identity'])
            c.call(lane,fork.remember_replay_boundary,device_uuid=spec['gpu_uuid_runtime'],model_sha256=spec['model_identity'])
            handoff=c.call(lane,fork.install_at_selected_query,tick=clock_start,device_uuid=spec['gpu_uuid_runtime'],model_sha256=spec['model_identity'])
            write(target/(branch+'_sampler_fork.json'),handoff)
        live=LiveFeedback(c,lane,mask_model,item['earliest_seed'],seed_masks,rows,ticks,item['instruction'],feedback_geometry,predictor,cache)
        if fork is not None:live.bind_fork(fork)
        prefix_meta=dict(prefix_physics_sha256=obs.physical(),prefix_rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()),prefix_rng_components=rng_components(),source_family='current_RGB_native')
        origin_actions=compose(selected['base'],selected['raw_delta'],branch)
        origin=capture_chunk(task,obs,initial_obs,origin_actions,max_ticks=5000,check_cancel=c.check_cancel,kind=kind,once_base=selected['base'],on_arrival=live.capture_callback('origin',clock_start,len(rows)-1))
        export_chunk(target,branch+'_origin',origin,planned_actions=origin_actions,prefix_meta=prefix_meta,absolute_start_tick=clock_start)
        status=dict(origin_available=origin['available'],origin_feedback_ready=bool(origin.get('feedback_ready',False)),following_query_sampled=False,following_available=False,delayed_query_sampled=False,delayed_available=False,native_terminal_before_following=False,native_terminal_before_delayed=False)
        if not origin['available']:
            status['origin_reason']=origin['reason'];live.export(target,branch);write(target/(branch+'_continuation.json'),status);return
        end=clock_start+len(origin['packet']['dt_s'])
        if fork is not None:fork.finish_origin(origin['commands_returned'],end_tick=end,requested95_arrived=any(r.get('requested95_tick') is not None for r in origin['completion']))
        live.close_origin(origin)
        if origin['feedback_ready'] and len(origin['packet']['dt_s'])<5000 and native_continue(task):
            current=current_at(origin['terminal'],end)
            op=lambda:query_current(base,head,robots,projection,current,item['instruction'],prev,decision_tick=end)
            if sequential:
                live.reference_boundary(task,current,query=len(rows))
                before=full_hash(recovery.rng_capture())
                following=c.call(lane,query_current,base,head,robots,projection,current,item['instruction'],prev,decision_tick=end)
                after=full_hash(recovery.rng_capture())
                from native_nextchunk_v1.sampler_fork import base_hash
                qm=dict(query=len(rows),rng_before=before,rng_after=after,base_sha256=base_hash(following['base']))
            else:
                following,qm=live.draw(task,current,op,query=len(rows),delayed=False,device_uuid=spec['gpu_uuid_runtime'],model_sha256=spec['model_identity'])
            save_query(target,branch+'_following_query',following,qm)
            status['following_query_sampled']=True
            fmeta=dict(prefix_physics_sha256=obs.physical(),prefix_rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()),prefix_rng_components=rng_components(),source_family='current_RGB_native',candidate_applied=False)
            result=capture_chunk(task,obs,current,following['base'],max_ticks=5000-len(origin['packet']['dt_s']),check_cancel=c.check_cancel,on_arrival=live.capture_callback('following',end,len(rows)))
            export_chunk(target,branch+'_following',result,planned_actions=following['base'],prefix_meta=fmeta,absolute_start_tick=end)
            status['following_available']=result['available']
            if fork is not None:fork.finish_following(result['commands_returned'],end_tick=end+len(result.get('packet',{}).get('dt_s',[])))
            used=len(origin['packet']['dt_s'])+len(result.get('packet',{}).get('dt_s',[]))
            following_complete=bool(result['available']and result['origin']['complete_origin']and result['commands_returned']==32)
            if following_complete and used<5000 and native_continue(task):
                next_tick=clock_start+used;now=current_at(result['terminal'],next_tick)
                op=lambda:query_current(base,head,robots,projection,now,item['instruction'],following['previous'],decision_tick=next_tick)
                if sequential:
                    live.reference_boundary(task,now,query=len(rows)+1)
                    before=full_hash(recovery.rng_capture())
                    delayed=c.call(lane,op);after=full_hash(recovery.rng_capture())
                    from native_nextchunk_v1.sampler_fork import base_hash
                    dm=dict(query=len(rows)+1,rng_before=before,rng_after=after,base_sha256=base_hash(delayed['base']))
                else:
                    delayed,dm=live.draw(task,now,op,query=len(rows)+1,delayed=True,device_uuid=spec['gpu_uuid_runtime'],model_sha256=spec['model_identity'])
                save_query(target,branch+'_delayed_query',delayed,dm);status['delayed_query_sampled']=True
                dmeta=dict(prefix_physics_sha256=obs.physical(),prefix_rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()),prefix_rng_components=rng_components(),source_family='current_RGB_native',candidate_applied=False)
                last=capture_chunk(task,obs,now,delayed['base'],max_ticks=5000-used,check_cancel=c.check_cancel,on_arrival=live.capture_callback('following',next_tick,len(rows)+1))
                export_chunk(target,branch+'_delayed',last,planned_actions=delayed['base'],prefix_meta=dmeta,absolute_start_tick=next_tick)
                status['delayed_available']=last['available']
                if fork is not None:fork.finish_delayed(last['commands_returned'],end_tick=next_tick+len(last.get('packet',{}).get('dt_s',[])))
            elif not native_continue(task):status['native_terminal_before_delayed']=True
        elif not native_continue(task):status['native_terminal_before_following']=True
        live.export(target,branch)
        write(target/(branch+'_continuation.json'),status)
    for branch in ('Z_reference','C_original','Z_sham'):
        try:run_branch(branch,folder)
        finally:task.close_env(clear_cache=False)
    if item.get('requires_sequential_reference'):
        ref=folder/'sequential_reference';ref.mkdir()
        checks=0
        for branch in ('Z_reference','C_original'):
            try:run_branch(branch,ref,sequential=True)
            finally:task.close_env(clear_cache=False)
            checks+=reference_equal(folder,ref,branch)
        write(folder/'sequential_reference_checks.json',dict(passed=True,checks=checks,
             source='Actual sequentialcurrentRGB queries andnativeexecution, no samplerfork or cache-skipped draws',branches=['Z_reference','C_original']))
