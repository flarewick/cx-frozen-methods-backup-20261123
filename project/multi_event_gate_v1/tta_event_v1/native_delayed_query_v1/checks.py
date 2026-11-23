"""Actual pinned noise/sample/infer bodies, fixture forward; no CUDA/simulator."""
from pathlib import Path
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import numpy as np
C = Path(__file__).resolve().parent
T = C.parent
sp = importlib.util.spec_from_file_location('delayed_pinned_sampler_fixture', T/'native_nextchunk_v1/sampler_checks.py')
s = importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
import torch
from native_delayed_query_v1.sampler import DelayedQueryFork
from native_nextchunk_v1.sampler_fork import base_hash
from native_delayed_query_v1.boundary import ShadowBoundary, boundary_proof, native_continue, image_sha
from native_phase_feedback_v1.feedback_queue import FeedbackQueue
O = T/'outputs/native_delayed_query_v1'

def run():
    router = s.RngRouter(cuda=False); router.install()
    checks = 0; records = []
    H = '1'*64
    def reject(fn):
        nonlocal checks
        before = s.full_hash(router.capture(router.current()))
        try:
            fn()
        except ValueError:
            assert before == s.full_hash(router.capture(router.current()))
            checks += 1
        else:
            raise AssertionError('Invalid control admitted')
    def obs(tick):
        return dict(observation_tick=tick, rgb=tuple(np.full((240,320,3), k, np.uint8) for k in (12,23,34)),
                    command_state=np.zeros(14, np.float32))
    def queue(early=False):
        q = FeedbackQueue(); first=obs(200); sha=image_sha(first['rgb'][0]); q.arrive(200,0.,sha); ticket=q.register(2)
        if early:
            q.arrive(225,.1,sha)
            q.arrive(300,.4,sha)
            q.window(ticket,row=0,start_tick=225,end_tick=300,kind='origin',command_start_tick=200,command_end_tick=300)
            pt, ps = 220, .08
        else:
            q.arrive(250,.2,sha); q.arrive(300,.4,sha)
            q.window(ticket,row=0,start_tick=250,end_tick=300,kind='origin',command_start_tick=200,command_end_tick=300)
            pt, ps = 290, .36
        q.close_origin(ticket,commands_returned=32,complete_native32=True,receipt_tick=300,
                       phase95_tick=pt,phase95_elapsed_s=ps)
        return q, ticket
    def source(lane):
        model=s.FixtureModel()
        fork=DelayedQueryFork(router,lane,model,device_uuid='CPU',model_sha256=s.MODEL_SHA)
        for query in range(3):
            fork.draw(s.operation(model,np.full(14,query*.01,np.float32)),query=query,tick=query*100)
            if query<2:
                s.simulate_native();fork.finish_prefix(32,end_tick=(query+1)*100)
        return fork,model
    def installed(fresh, lane, name):
        s.simulate_native();s.simulate_native()
        fork=fresh.branch(name,router,lane,device_uuid='CPU',model_sha256=s.MODEL_SHA)
        fork.remember_replay_boundary(device_uuid='CPU',model_sha256=s.MODEL_SHA)
        fork.install_at_selected_query(tick=200,device_uuid='CPU',model_sha256=s.MODEL_SHA)
        return fork
    try:
        discovery=s.Lane('delayed_discovery');s.seed(router,discovery)
        with router.bind(discovery),router.torch_scope(discovery):
            fresh, model=source(discovery)
        for early in (False,True):
            for name,effect in (('Z_reference',0),('C_original',1),('Z_sham',0)):
                name=name+('_early'if early else '_delayed')
                ref=s.Lane('sequential_'+name);s.seed(router,ref)
                with router.bind(ref),router.torch_scope(ref):
                    rm=s.FixtureModel()
                    for query in range(3):
                        selected=s.operation(rm,np.full(14,query*.01,np.float32))()['base']
                        if query<2:s.simulate_native()
                    selected_hash=s.full_hash(router.capture(ref))
                    s.simulate_native(33 if effect else 32)
                    origin_hash=s.full_hash(router.capture(ref))
                    first=s.operation(rm,np.full(14,.03+effect*.01,np.float32))()['base']
                    first_hash=s.full_hash(router.capture(ref))
                    s.simulate_native(34 if effect else 32)
                    following_hash=s.full_hash(router.capture(ref))
                    second=s.operation(rm,np.full(14,.04+effect*.01,np.float32))()['base']
                    second_hash=s.full_hash(router.capture(ref))
                    s.simulate_native()
                    final_hash=s.full_hash(router.capture(ref))
                lane=s.Lane(name);s.seed(router,lane)
                with router.bind(lane),router.torch_scope(lane):
                    fork=installed(fresh,lane,name)
                    assert s.full_hash(router.capture(lane))==selected_hash;checks+=1
                    assert base_hash(selected)==fresh.ledger.records[-1]['base_sha256'];checks+=1
                    s.simulate_native(33 if effect else 32)
                    fork.finish_origin(32,end_tick=300,requested95_arrived=True)
                    assert s.full_hash(router.capture(lane))==origin_hash;checks+=1
                    q,ticket=queue(early)
                    ctl=ShadowBoundary(fork,q)
                    task=SimpleNamespace(take_action_cnt=96,step_lim=1000,eval_success=False)
                    reject(lambda:ctl.draw(task,obs(299),s.operation(model),query=3,tick=300,
                           delayed=False,device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    one,note1,packet1=ctl.draw(task,obs(300),s.operation(model,np.full(14,.03+effect*.01,np.float32)),
                           query=3,tick=300,delayed=False,device_uuid='CPU',model_sha256=s.MODEL_SHA)
                    assert np.array_equal(one['base'],first)and s.full_hash(router.capture(lane))==first_hash;checks+=1
                    assert (packet1 is not None)==early;checks+=1
                    reject(lambda:fork.draw_delayed(s.operation(model),query=4,tick=400,
                           device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    s.simulate_native(34 if effect else 32);fork.finish_following(32,end_tick=400)
                    assert s.full_hash(router.capture(lane))==following_hash;checks+=1
                    q.arrive(400,.8,image_sha(obs(400)['rgb'][0]))
                    accepted=q.window(ticket,row=1,start_tick=300,end_tick=400,kind='following',
                           command_start_tick=300,command_end_tick=400)
                    assert accepted != early;checks+=1
                    task.take_action_cnt=128
                    two,note2,packet2=ctl.draw(task,obs(400),s.operation(model,np.full(14,.04+effect*.01,np.float32)),
                           query=4,tick=400,delayed=True,device_uuid='CPU',model_sha256=s.MODEL_SHA)
                    assert np.array_equal(two['base'],second)and s.full_hash(router.capture(lane))==second_hash;checks+=1
                    assert (packet2 is None)==early;checks+=1
                    assert q.released==1 and (packet1 or packet2).rows==((0,)if early else(1,));checks+=1
                    proof,digest=boundary_proof(obs(400),query=4,tick=400)
                    assert ctl.records[1]['boundary']==proof and ctl.records[1]['boundary_sha256']==digest;checks+=1
                    assert len(fork.ledger.records)==5 and all(r['returned']==32 for r in fork.ledger.records[:4]);checks+=1
                    s.simulate_native();fork.finish_delayed(32,end_tick=500)
                    assert s.full_hash(router.capture(lane))==final_hash;checks+=1
                    reject(lambda:fork.draw_delayed(s.operation(model),query=5,tick=500,device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    reject(lambda:fork.draw_following(s.operation(model),query=5,tick=500,device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    reject(lambda:fork.install_at_selected_query(tick=500,device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    assert ctl.close()['released_feedback_packets']==1;checks+=1
                    reject(lambda:ctl.draw(task,obs(500),s.operation(model),query=5,tick=500,
                           delayed=True,device_uuid='CPU',model_sha256=s.MODEL_SHA))
                    records.append(dict(branch=name,first_and_delayed_base_bitwise=True,all_CPU_RNG_exact=True,
                            released_query=(packet1 or packet2).query,internal_noise_calls_per_query=1))
        # Native last-action terminal flags stop BEFORE queue consumption or draw.
        for terminal,step_limit in ((True,1000),(False,128)):
            lane=s.Lane('terminal_'+str(terminal));s.seed(router,lane)
            with router.bind(lane),router.torch_scope(lane):
                own,model=source(lane)
                # Branch starts from its selected query without replaying prefix.
                fork=own.branch('terminal',router,lane,device_uuid='CPU',model_sha256=s.MODEL_SHA)
                fork.remember_replay_boundary(device_uuid='CPU',model_sha256=s.MODEL_SHA)
                fork.install_at_selected_query(tick=200,device_uuid='CPU',model_sha256=s.MODEL_SHA)
                s.simulate_native();fork.finish_origin(32,end_tick=300,requested95_arrived=True)
                fork.draw_following(s.operation(model),query=3,tick=300,device_uuid='CPU',model_sha256=s.MODEL_SHA)
                s.simulate_native();fork.finish_following(32,end_tick=400)
                q,ticket=queue(False);q.at_query(query=3,tick=300,query_proof_sha256=H)
                q.arrive(400,.8,image_sha(obs(400)['rgb'][0]))
                q.window(ticket,row=1,start_tick=300,end_tick=400,kind='following',command_start_tick=300,command_end_tick=400)
                assert q.eligible(ticket)==(1,);checks+=1
                ctl=ShadowBoundary(fork,q);before=s.full_hash(router.capture(lane))
                task=SimpleNamespace(take_action_cnt=128,step_lim=step_limit,eval_success=terminal)
                def forbidden():raise AssertionError('No query after native termination')
                assert ctl.draw(task,obs(400),forbidden,query=4,tick=400,delayed=True,
                        device_uuid='CPU',model_sha256=s.MODEL_SHA)is None;checks+=1
                assert s.full_hash(router.capture(lane))==before and fork.delayed_draws==0;checks+=1
                assert q.released==0 and ctl.final['pending_sources']==1 and ctl.closed;checks+=1
        # Malformed native state cannot enable a query.
        for count,limit,done in ((0,None,False),(True,100,False),(0,100,1),(-1,100,False)):
            try:native_continue(SimpleNamespace(take_action_cnt=count,step_lim=limit,eval_success=done))
            except ValueError:checks+=1
            else:raise AssertionError('Bad evaluator state admitted')
        assert not torch.cuda.is_initialized();checks+=1
    finally:router.uninstall()
    report=dict(passed=True,checks=checks,branches=records,
        original_noise_sample_infer_method_AST_sha256=[s.h1,s.h2,s.h3],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (
           C/'sampler.py',C/'boundary.py',C/'checks.py',
           T/'native_nextchunk_v1/sampler_checks.py',T/'native_nextchunk_v1/sampler_current.py',
           T/'native_nextchunk_v1/sampler_fork.py',T/'native_nextchunk_v1/query_ledger.py',
           T/'native_phase_feedback_v1/feedback_queue.py')},
        scope='Actual pinned sampler method bodies and real CPU RngRouter with fixture forward/observations/native steps; not full VLA, real simulator, CUDA, online tracker or TTA benefit.',
        first_and_second_natural_query_exact=True,one_packet_per_source=True,
        terminal_after_last_returned32_causes_no_draw_and_no_feedback=True,
        online_optimizer_updates=0,real_GPU_ready=False,TTA_utility_proven=False)
    O.mkdir(parents=True,exist_ok=True);p=O/'sampler_boundary_cpu_checks.json'
    encoded=json.dumps(report,indent=2)+'\n'
    if p.exists():assert p.read_text()==encoded
    else:p.write_text(encoded)
    print('NATURAL_DELAYED_QUERY_REAL_SAMPLER_BODY_CPU_PASS',checks,'NO_SIMULATOR_OR_GPU_CLAIM',flush=True)
if __name__=='__main__':run()
