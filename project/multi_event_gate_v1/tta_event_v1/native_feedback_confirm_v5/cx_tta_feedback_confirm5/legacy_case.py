"""Four independent CPU scene lanes; one owner computes frozen batch3 features."""
import argparse,atexit,importlib,json,os,queue,signal,threading,time,traceback,uuid
from pathlib import Path
import numpy as np
from .paths import PROJECT,OUTPUT,read,write,output,verify,digest
from .contract import BRANCHES,TICKS,case_branches,dataset,load_journal,commit,validate_packet
from .views import VIEWS,batched_encode
from .runtime import Observer
from .throughput import TaskPool,replay_prefix,discard_observation,assert_original_parity
from .lanes import Coordinator,Lane,gripper_only
from cx_tta_clock.clock import ACTIVE,Clock,ClockComplete
from actuation_telemetry import Recorder,ACTIVE as TRACE
from .schedule import instrument,ACTIVE as SCHEDULE,Plan
from cx_tta_clock.runtime import rng_components


def run_legacy(task,c,cfg,item,spec,index,folder,lane,base,head,robots,projection,observer,routed_seed,note,query,compose):
    from cx_tta import recovery_contract as recovery
    from cx_tta.eval_diagnostics import array_hash
    from cx_tta_clock.runtime import rng_components
    from .prefix_time import Counter
    # Exactly one discovery pass. Models receive only legal current inputs.
    from cx_tta.multi_event import EventDetector,EventConfig
    from cx_tta.empirical_onset_gate import candidate_packet
    detector=EventDetector(EventConfig(**spec['event_config']))
    prefix=[];previous=None;found=None;query_rows=[];query_data=[];unavailable=None;prefix_total=0;query_ticks=[];clock_chunks=[];prefix_dt=None
    try:
        discovery=observer();current=replay_prefix(task,item,[],cfg,discovery.read,routed_seed)
        query_start=item.get('coverage_start_query',0);maximum=query_start+4
        prefix_dt=float(task.scene.get_timestep())
        for q in range(maximum):
            query_ticks.append(prefix_total)
            note('discover',lane=lane.name,case=item['case_id'],query=q)
            before_rng=recovery.rng_fingerprint(recovery.rng_capture())
            proposal=c.call(lane,query,base,head,robots,projection,current,discovery.anchor[0],item['instruction'],previous)
            after_rng=recovery.rng_fingerprint(recovery.rng_capture())
            query_data.append(dict(rgb=tuple(im.copy() for im in current['rgb']),images=dict(current['images']),command_state=current['command_state'].copy(),base=proposal['base'].copy()))
            if before_rng==after_rng:raise ValueError('Sole internal-noise sampler failed to advance RNG')
            event,evidence=detector.step(q,candidate_packet(proposal['legal'],proposal['raw_delta']))
            query_rows.append(dict(query=q,event=event,rng_before=before_rng,rng_after=after_rng))
            previous=proposal['previous']
            if event is not None and q>=query_start:
                found=proposal;discovery_initial=current;discovery_physics=discovery.physical();break
            if q==maximum-1:unavailable='no_event_within_four_queries';break
            start_tick=prefix_total
            with Counter(float(task.scene.get_timestep())) as counter:
                for command in proposal['base']:
                    c.check_cancel();before_count=task.take_action_cnt;task.take_action(command);discard_observation(task)
                    if task.take_action_cnt!=before_count+1:
                        unavailable='native_prefix_censored';break
            if float(task.scene.get_timestep())!=counter.dt or counter.dt!=prefix_dt:raise ValueError('Nativeprefix dt changed')
            prefix_total+=counter.ticks
            clock_chunks.append(dict(query=q,start_tick=start_tick,end_tick=prefix_total,censored=bool(unavailable)))
            if unavailable:break
            prefix.append(proposal['base'].copy());discovery=observer();current=discovery.read()
    finally:task.close_env(clear_cache=False)
    write(folder/'discovery.json',dict(queries=query_rows,sampler_calls=len(query_rows),prefix_chunks=len(prefix),
        no_expert_screening=True,uses_outcome_labels=False,scope='New eager frozen-policy source, not historical compiled-policy equality'))
    from .legacy_history import save_history
    save_history(folder,query_data)
    write(folder/'prefix_clock.json',dict(query_observation_ticks=query_ticks,executed_chunks=clock_chunks,dt_s=prefix_dt,physical_interval_duration_available=True,native_returned_counts_and_actual_slow_steps=True,semantics='Actualcounts for ownbaseprefix; RGBhistoryduration flagFalse islegacy schema, exacttiming isthissidecar.'))
    if unavailable:
        write(folder/'unavailable.json',dict(reason=unavailable,effect_used=False))
        return
    base_chunk=found['base'];delta=found['raw_delta']
    reference=Path(spec['bridge_references'][item['case_id']])
    with np.load(reference/'proposal.npz',allow_pickle=False) as z:
        if not np.array_equal(base_chunk,z['base']) or not np.array_equal(delta,z['raw_delta']) or not np.array_equal(np.stack(prefix) if prefix else np.zeros((0,32,14),np.float32),z['prefix']):raise ValueError('Rediscovered legal proposal/prefix differs from immutable developer source')
    with (folder/'proposal.npz').open('xb') as f:np.savez_compressed(f,base=base_chunk,raw_delta=delta,
        prefix=np.stack(prefix) if prefix else np.zeros((0,32,14),np.float32))
    with (folder/'discovery_anchor.npz').open('xb') as f:np.savez_compressed(f,**{k:discovery_initial[k] for k in ('joint','visual','command_state')})
    item=dict(item,fork_query=len(prefix))
    for branch in case_branches(folder):
        c.check_cancel();note('replay',lane=lane.name,case=item['case_id'],branch=branch)
        obs=observer();began=time.monotonic()
        try:
            initial_obs=replay_prefix(task,item,prefix,cfg,obs.read,routed_seed)
            prefix_seconds=time.monotonic()-began
            if obs.physical()!=discovery_physics:raise ValueError('Discovery/replayed prefix physics differs')
            for key in ('joint','visual','command_state'):
                if not np.array_equal(initial_obs[key],discovery_initial[key]):raise ValueError('Discovery/replayed prefix observable differs: '+key)
            prefix_physics=obs.physical();prefix_rng=recovery.rng_fingerprint(recovery.rng_capture());components=rng_components()
            for key in ('joint','visual'):
                if any(not np.array_equal(initial_obs[key][0],initial_obs[key][i]) for i in (1,2)):raise ValueError('Equal-input batch differs')
            from native_stage_v1.clock import StageClock,OriginRecorder,COMMON_TICKS
            from stage_feedback_v2.timing import FeedbackQueue
            from stage_feedback_bridge_v1.bridge import AfterActuation,ACTIVE as FEEDBACK
            from cx_tta.eval_diagnostics import json_hash
            limit=500 if index<spec['compatibility_count'] else 5000
            clock=StageClock(max_ticks=limit);token=ACTIVE.set(clock)
            recorder=OriginRecorder(max_ticks=limit);trace_token=TRACE.set(recorder)
            queue_feedback=FeedbackQueue();event=query_rows[-1]['event']
            if len(event['kinds'])!=1:raise ValueError('Explicit unsupported multikind event')
            queue_feedback.register('event','origin',event['kinds'][0],initial_obs['command_state'][[6,13]],json_hash(initial_obs['images']),decision_tick=0)
            def sample(tick,elapsed):
                value=obs.read();value.update(tick=tick,elapsed_s=elapsed,physics_sha256=obs.physical(),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()),frame_id=None)
                value['frame_id']=json_hash(value['images']);return value
            hook=AfterActuation(recorder,queue_feedback,'origin',sample,common_ticks=[t for t in COMMON_TICKS if t<=limit],expected_stop=ClockComplete)
            feedback_token=FEEDBACK.set(hook);schedule_token=SCHEDULE.set(None);returned=0;native_censored=False;horizon=False
            try:
                for command in compose(base_chunk,delta,branch):
                    c.check_cancel();before_count=task.take_action_cnt;task.take_action(command);discard_observation(task)
                    if task.take_action_cnt!=before_count+1:native_censored=True;break
                    returned+=1
            except ClockComplete:horizon=True
            finally:
                SCHEDULE.reset(schedule_token);FEEDBACK.reset(feedback_token);TRACE.reset(trace_token);ACTIVE.reset(token)
            if not recorder.rows:
                write(folder/'unavailable.json',dict(reason='zero_native_steps',branch=branch,effect_used=False));break
            if not hook.readouts or hook.readouts[-1]['tick']!=len(clock.dt):
                tick=len(clock.dt);value=sample(tick,float(sum(clock.dt)));requests=queue_feedback.observe(tick,decision_tick=tick,frame_id=value['frame_id'])
                hook.readouts.append(dict(tick=tick,elapsed_s=float(sum(clock.dt)),common_time_readout=False,stage_requests=requests,observation=value))
            reason='measurement_horizon_end' if horizon else 'native_command_censored' if native_censored else 'origin_chunk_completed'
            recorder.seal(reason,commands_returned=returned)
            if reason=='origin_chunk_completed':queue_feedback.end_chunk('origin',arrived_tick=len(clock.dt))
            else:queue_feedback.censor('capture_horizon_end' if horizon else 'native_clock_censored',arrived_tick=len(clock.dt))
            clock.samples=[dict(v['observation'],phase_requests=v['stage_requests'],common_time_readout=v['common_time_readout']) for v in hook.readouts]
            ticks=tuple(v['tick'] for v in clock.samples)
            packet=dict(ticks=np.array(ticks,np.int64),elapsed_s=np.array([v['elapsed_s'] for v in clock.samples],np.float64),
                joint=np.stack([v['joint'] for v in clock.samples]),visual=np.stack([v['visual'] for v in clock.samples]),
                command_state=np.stack([v['command_state'] for v in clock.samples]),commands=np.asarray(clock.commands,np.float32),dt_s=np.array(clock.dt,np.float64))
            validate_packet(packet,ticks);trace_file=folder/(branch+'_actuation.npz')
            with trace_file.open('xb') as f:np.savez_compressed(f,**recorder.packet())
            write(folder/(branch+'_phase.json'),dict(receipt=queue_feedback.receipt(),origin=recorder.sealed,
                readouts=[dict(tick=v['tick'],elapsed_s=v['elapsed_s'],common_time_readout=v['common_time_readout'],stage_requests=v['stage_requests']) for v in hook.readouts],semantic_mask_source='Cached initial RGB/language mask hashes must match before downstream use; no GT'))
            with (folder/(branch+'_rgb.npz')).open('xb') as rgbfile:np.savez_compressed(rgbfile,**{**{'initial_'+str(i):initial_obs['rgb'][i] for i in range(3)},**{'samples_'+str(i):np.stack([v['rgb'][i] for v in clock.samples]) for i in range(3)}})
            file=folder/(branch+'.npz')
            with file.open('xb') as f:np.savez_compressed(f,**packet)
            meta=dict(planned_chunk_sha256=array_hash(compose(base_chunk,delta,branch)),actuation_sha256=digest(trace_file),view_order=list(VIEWS),file_sha256=digest(file),prefix_physics_sha256=prefix_physics,
                prefix_rng_sha256=prefix_rng,prefix_rng_components=components,
                initial={k:array_hash(initial_obs[k]) for k in ('joint','visual','command_state')},initial_images=initial_obs['images'],initial_tokens=initial_obs['token_sha256'],
                samples=[{k:s[k] for k in ('tick','elapsed_s','physics_sha256','rng_sha256','view_images','token_sha256')} for s in clock.samples],
                native_actions_started=int(task.take_action_cnt)-32*q,complete_episode=False,outcomes_used=False,
                wall_seconds=time.monotonic()-began,prefix_wall_seconds=prefix_seconds,encoder_wall_seconds=obs.encoder_seconds,
                observation_wall_seconds=obs.observation_seconds,render_upload=dict(task.cameras._cx_pose_upload_stats),lane=lane.name)
            write(folder/(branch+'.json'),meta)
            if branch=='Z_reference':
                with (folder/'initial.npz').open('xb') as f:np.savez_compressed(f,**{k:initial_obs[k] for k in ('joint','visual','command_state')})
        finally:task.close_env(clear_cache=False)
    if item['case_id'] in spec['bridge_references']:
        from .continuation import parity
        write(folder/'prefix_compatibility.json',parity(folder,Path(spec['bridge_references'][item['case_id']])))

    if item.get('legacy_full32'):
        from .full_legacy import full_parity
        write(folder/'full_legacy_compatibility.json',full_parity(folder,Path(spec['full_legacy_references'][item['case_id']])))
