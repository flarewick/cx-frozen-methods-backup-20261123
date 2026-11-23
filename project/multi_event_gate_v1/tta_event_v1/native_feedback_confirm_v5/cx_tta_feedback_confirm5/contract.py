"""Separate old compatibility and true current-RGB two-chunk developer records."""
from pathlib import Path
import numpy as np
from .paths import read,write,output,digest,identity
from native_stage_v1.cx_tta_stage.contract import validate_packet,load_packet,same
BRANCHES=('Z_reference','C_original','Z_sham')
TICKS=(32,64,125,187,250,375,500)
def case_branches(folder):return BRANCHES

def verify_segment(folder,name):
    from cx_tta.eval_diagnostics import array_hash
    packet=load_packet(folder/(name+'.npz'));validate_packet(packet)
    meta=read(folder/(name+'.json'));trace=load_packet(folder/(name+'_actuation.npz'))
    receipt=read(folder/(name+'_receipt.json'));rgb=load_packet(folder/(name+'_rgb.npz'))
    assert digest(folder/(name+'.npz'))==meta['file_sha256']
    assert digest(folder/(name+'_actuation.npz'))==meta['actuation_sha256']
    n=len(packet['dt_s']);frames=len(packet['ticks'])
    assert np.array_equal(trace['dt_s'],packet['dt_s'])
    assert trace['arm_position_target'].shape==trace['arm_velocity_target'].shape==(n,12)
    assert trace['gripper_logical_command'].shape==(n,2)
    for k in ('gripper_position_target','gripper_velocity_target','gripper_joint_counts'):assert len(trace[k])==n and np.isfinite(trace[k]).all()
    assert np.array_equal(trace['arm_position_target'].astype(np.float32),packet['commands'][:,[0,1,2,3,4,5,7,8,9,10,11,12]])
    assert np.array_equal(trace['gripper_logical_command'].astype(np.float32),packet['commands'][:,[6,13]])
    assert receipt['origin']['committed_ticks']==n and len(receipt['readouts'])==frames
    assert receipt['commands_returned']==receipt['origin']['commands_returned']
    assert 0<=receipt['commands_returned']<=32
    if receipt['origin']['complete_origin']:assert receipt['commands_returned']==32
    for cam in range(3):
        assert rgb['initial_'+str(cam)].dtype==np.uint8
        assert rgb['samples_'+str(cam)].shape==(frames,*rgb['initial_'+str(cam)].shape)
        assert array_hash(rgb['initial_'+str(cam)])==meta['initial_images'][str(cam)]
        for i,row in enumerate(meta['samples']):
            assert row['tick']==int(packet['ticks'][i])
            assert array_hash(rgb['samples_'+str(cam)][i])==row['view_images'][0][str(cam)]
    assert meta['local_ticks_not_global'] and not meta['complete_episode'] and not meta['outcomes_used']
    return packet,trace,rgb,meta,receipt

def verify_case(folder,projection=None):
    folder=Path(folder);case=read(folder/'case.json')
    if case['source_family']=='legacy_anchor_compatibility':
        from native_stage_v1.cx_tta_stage.contract import verify_case as legacy_verify
        result=legacy_verify(folder,projection)
        if case['legacy_full32']:
            assert read(folder/'full_legacy_compatibility.json')['passed']
        return dict(result,source_family=case['source_family'],excluded_from_current_data=True)
    assert case['source_family']=='current_RGB_native'
    if (folder/'unavailable.json').exists():
        row=read(folder/'unavailable.json')
        assert row['reason'] in ('no_event_within_fixed_budget','native_prefix_censored','unsupported_multiple_event_kinds','unsupported_requested_excursion','zero_native_steps','native_episode_terminated_before_event')
        return dict(available=False,source_family=case['source_family'],reason=row['reason'],old_developer_retained=case.get('role')!='independent_TRAIN_confirmation',independent_case_retained=case.get('role')=='independent_TRAIN_confirmation')
    from cx_tta.eval_diagnostics import array_hash
    from .query import compose
    proposal=load_packet(folder/'proposal.npz');initial=load_packet(folder/'initial_current.npz')
    qhistory=load_packet(folder/'observed_history.npz');hmeta=read(folder/'actor_history.json')
    discovery=read(folder/'discovery.json');clock=read(folder/'prefix_clock.json')
    assert len(qhistory['queries'])==len(discovery['queries'])==len(hmeta['queries'])==len(clock['query_observation_ticks'])
    assert len(proposal['prefix'])+1==len(qhistory['queries'])
    assert np.array_equal(proposal['prefix'],qhistory['base_chunks'][:-1])
    assert np.array_equal(proposal['base'],qhistory['base_chunks'][-1])
    assert np.array_equal(qhistory['legal_base'],qhistory['base_chunks'])
    scaled_state=qhistory['states'].astype(np.float32).copy()
    scaled_state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
    assert np.array_equal(qhistory['legal_state'],scaled_state)
    assert np.array_equal(qhistory['legal_past_state'][1:],scaled_state[:-1])
    assert not qhistory['legal_has_history'][0].any() and qhistory['legal_has_history'][1:].all()
    assert len(discovery['sampler_witnesses'])==len(qhistory['queries'])
    for q,note in enumerate(discovery['sampler_witnesses']):
        assert note['query']==q and note['tick']==clock['query_observation_ticks'][q]
        assert note['rng_before']==discovery['queries'][q]['rng_before'] and note['rng_after']==discovery['queries'][q]['rng_after']
        assert note['shape']==[1,32,32] and note['before']!=note['after']
        assert note['environment_before']==note['environment_after']
    for q,row in enumerate(hmeta['queries']):
        assert row['query']==q and row['actor_observation_tick']==clock['query_observation_ticks'][q]
        assert row['actor_image_source']=='actual_current_RGB_not_initial_anchor'
        for cam in range(3):assert array_hash(qhistory['cam'+str(cam)][q])==row['actor_images'][str(cam)]
    data={};statuses={}
    for branch in BRANCHES:
        status=read(folder/(branch+'_continuation.json'));statuses[branch]=status
        if not status['origin_available']:
            assert status['origin_reason']=='zero_native_steps';data[branch]=None;continue
        p,t,r,m,receipt=verify_segment(folder,branch+'_origin')
        assert m['planned_chunk_sha256']==array_hash(compose(proposal['base'],proposal['raw_delta'],branch))
        for k in initial:assert array_hash(initial[k])==m['initial'][k]
        ready=receipt['feedback_ready']
        if ready:
            row=receipt['completion'][0]
            assert row['status']=='feedback_ready' and receipt['commands_returned']==32
            assert row['requested95_tick']is not None and row['requested95_tick']<=len(p['dt_s'])
            assert row['origin_completed_tick']==row['observation_tick']==len(p['dt_s'])
        assert status['origin_feedback_ready']==ready
        follow=None
        if status['following_query_sampled']:
            assert ready and len(p['dt_s'])<5000
            q=load_packet(folder/(branch+'_following_query.npz'));qm=read(folder/(branch+'_following_query.json'))
            assert qm['actor_observation_tick']==m['absolute_start_tick']+len(p['dt_s'])
            assert qm['query']==len(qhistory['queries'])
            for cam in range(3):assert array_hash(r['samples_'+str(cam)][-1])==qm['actor_images'][str(cam)]
            if status['following_available']:
                follow=verify_segment(folder,branch+'_following')
                assert follow[3]['planned_chunk_sha256']==array_hash(q['base'])
                assert follow[3]['absolute_start_tick']==qm['actor_observation_tick']
                assert len(p['dt_s'])+len(follow[0]['dt_s'])<=5000
                assert not follow[4]['completion'] and not follow[4]['feedback_ready']
            else:assert read(folder/(branch+'_following_unavailable.json'))['reason']=='zero_native_steps'
        else:assert not ready or len(p['dt_s'])>=5000 or status['native_terminal_before_following']
        delayed=None
        if status['delayed_query_sampled']:
            assert follow is not None and follow[4]['commands_returned']==32 and follow[4]['origin']['complete_origin']
            query=load_packet(folder/(branch+'_delayed_query.npz'));qm=read(folder/(branch+'_delayed_query.json'))
            assert qm['query']==len(qhistory['queries'])+1
            assert qm['actor_observation_tick']==follow[3]['absolute_start_tick']+len(follow[0]['dt_s'])
            for cam in range(3):assert array_hash(follow[2]['samples_'+str(cam)][-1])==qm['actor_images'][str(cam)]
            if status['delayed_available']:
                delayed=verify_segment(folder,branch+'_delayed')
                assert delayed[3]['planned_chunk_sha256']==array_hash(query['base'])
                assert delayed[3]['absolute_start_tick']==qm['actor_observation_tick']
                assert len(p['dt_s'])+len(follow[0]['dt_s'])+len(delayed[0]['dt_s'])<=5000
                assert not delayed[4]['completion']and not delayed[4]['feedback_ready']
        data[branch]=(p,t,r,m,receipt,follow,delayed)
    a,b=data['Z_reference'],data['Z_sham']
    assert (a is None)==(b is None)
    if a is not None:
        for i in (0,1,2):assert same(a[i],b[i])
        assert a[3]['samples']==b[3]['samples'] and a[4]==b[4]
        assert (a[5]is None)==(b[5]is None)
        if a[5]is not None:
            for i in (0,1,2):assert same(a[5][i],b[5][i])
            assert a[5][3]['samples']==b[5][3]['samples'] and a[5][4]==b[5][4]
        if statuses['Z_reference']['following_query_sampled']:
            x=load_packet(folder/'Z_reference_following_query.npz');y=load_packet(folder/'Z_sham_following_query.npz')
            assert same(x,y)
            x,y=[read(folder/(v+'_following_query.json')) for v in ('Z_reference','Z_sham')]
            for k in ('actor_images','actor_observation_tick','rng_before','rng_after','base_sha256'):assert x[k]==y[k]
    if a is not None:
        assert(a[6]is None)==(b[6]is None)
        if a[6]is not None:
            for i in(0,1,2):assert same(a[6][i],b[6][i])
            assert a[6][3]['samples']==b[6][3]['samples']and a[6][4]==b[6][4]
        if statuses['Z_reference']['delayed_query_sampled']:
            assert statuses['Z_sham']['delayed_query_sampled']
            assert same(load_packet(folder/'Z_reference_delayed_query.npz'),load_packet(folder/'Z_sham_delayed_query.npz'))
            x,y=[read(folder/(v+'_delayed_query.json'))for v in('Z_reference','Z_sham')]
            for key in('actor_images','actor_observation_tick','rng_before','rng_after','base_sha256'):assert x[key]==y[key]
    if case.get('requires_sequential_reference'):
        assert read(folder/'sequential_reference_checks.json')['passed']
    return dict(available=True,source_family=case['source_family'],old_developer=True,online_updates=0,
        actual_current_RGB=True,once_sampler_fork=True,stage_complete_before_next_query=True,
        same_prefix=True,exact_three_chunk_Zsham=True,
        sequential_reference_passed=bool(case.get('requires_sequential_reference')),
        TTA_utility_proven=False)

def dataset(spec,gpu_uuid):
    from .paths import OUTPUT
    m=dict(protocol_identity=spec['identity'],gpu_uuid=gpu_uuid,encoder_batch=3,scene_order=[v['case_id'] for v in spec['scenes']])
    m['identity']=identity(m);root=output(OUTPUT/'collections'/m['identity'][:24]);root.mkdir(parents=True,exist_ok=True)
    f=root/'manifest.json'
    if f.exists():assert read(f)==m
    else:write(f,m)
    return root,m

def load_journal(root,manifest,spec):
    f=root/'journal.json';j=read(f) if f.exists() else dict(identity=manifest['identity'],cases=[])
    assert j['identity']==manifest['identity'] and len(j['cases'])<=len(spec['scenes'])
    for i,r in enumerate(j['cases']):
        assert r['case_id']==spec['scenes'][i]['case_id']
        for name,h in r['evidence_sha256'].items():assert digest(output(root/name))==h
        assert verify_case(root/r['folder'],spec['projection'])==r['audit']
    return j

def commit(root,manifest,spec,j,folder,session):
    audit=verify_case(folder,spec['projection']);i=len(j['cases'])
    r=dict(case_id=spec['scenes'][i]['case_id'],folder=str(folder.relative_to(root)),session=session,audit=audit,
        evidence_sha256={str(p.relative_to(root)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()})
    write(root/'journal.json',dict(identity=manifest['identity'],cases=[*j['cases'],r]));j['cases'].append(r);return r
