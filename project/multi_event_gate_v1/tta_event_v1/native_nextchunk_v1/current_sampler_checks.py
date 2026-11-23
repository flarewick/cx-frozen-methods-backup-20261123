"""Correct selected-query fork against unmodified sequential CPU method bodies.
Fixture forward only; actual current-RGB query API is tested separately.
"""
from sampler_checks import *
from sampler_current import CurrentQueryFork
def run_current():
    router=RngRouter(cuda=False);router.install();checks=0;records=[]
    try:
        discovery=Lane('current_discovery');seed(router,discovery);model=FixtureModel()
        with router.bind(discovery),router.torch_scope(discovery):
            source=CurrentQueryFork(router,discovery,model,device_uuid='CPU',model_sha256=MODEL_SHA)
            for q in range(3):
                source.draw(operation(model,np.full(14,q*.01,np.float32)),query=q,tick=q*100)
                if q<2:
                    simulate_native();source.finish_prefix(32,end_tick=(q+1)*100)
        for branch,effects in [('Z_reference',0),('C_original',1),('Z_sham',0)]:
            ref=Lane('true_sequence_'+branch);seed(router,ref)
            with router.bind(ref),router.torch_scope(ref):
                rm=FixtureModel()
                for q in range(3):
                    original=operation(rm,np.full(14,q*.01,np.float32))()['base']
                    if q<2:simulate_native()
                reference_at_selected=full_hash(router.capture(ref))
                simulate_native(33 if effects else 32)
                reference_after_origin=full_hash(router.capture(ref))
                expected=operation(rm,np.full(14,.03+effects*.01,np.float32))()['base']
                reference_after_next=full_hash(router.capture(ref))
            lane=Lane(branch);seed(router,lane)
            with router.bind(lane),router.torch_scope(lane):
                simulate_native();simulate_native()
                fork=source.branch(branch,router,lane,device_uuid='CPU',model_sha256=MODEL_SHA)
                rejected(lambda:fork.finish_origin(32,end_tick=300,requested95_arrived=True));checks+=1
                fork.remember_replay_boundary(device_uuid='CPU',model_sha256=MODEL_SHA)
                rejected(lambda:fork.install_at_selected_query(tick=201,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                handoff=fork.install_at_selected_query(tick=200,device_uuid='CPU',model_sha256=MODEL_SHA)
                assert full_hash(router.capture(lane))==reference_at_selected;checks+=1
                assert source.ledger.records[-1]['base_sha256']==hashlib.sha256(original.tobytes()).hexdigest();checks+=1
                simulate_native(33 if effects else 32)
                fork.finish_origin(32,end_tick=300,requested95_arrived=True)
                assert full_hash(router.capture(lane))==reference_after_origin;checks+=1
                result,note=fork.draw_following(operation(model,np.full(14,.03+effects*.01,np.float32)),
                   query=3,tick=300,device_uuid='CPU',model_sha256=MODEL_SHA)
                assert result['base'].tobytes()==expected.tobytes();checks+=1
                assert full_hash(router.capture(lane))==reference_after_next;checks+=1
                fork.finish_following(32,end_tick=400)
                rejected(lambda:fork.draw_following(operation(model),query=4,tick=400,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                rejected(lambda:fork.install(tick=400,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                records.append(dict(branch=branch,selected_original_next_all_CPU_RNG_exact=True,following_base_exact=True,handoff=handoff,note=note))
        for mode in ('env_mismatch','censor','no95','wrong_model','duplicate_install'):
            lane=Lane(mode);seed(router,lane)
            with router.bind(lane),router.torch_scope(lane):
                m=FixtureModel();s=CurrentQueryFork(router,lane,m,device_uuid='CPU',model_sha256=MODEL_SHA)
                s.draw(operation(m),query=0,tick=0)
                fork=s.branch(mode,router,lane,device_uuid='CPU',model_sha256=MODEL_SHA)
                if mode=='env_mismatch':np.random.random()
                fork.remember_replay_boundary(device_uuid='CPU',model_sha256=MODEL_SHA)
                if mode in ('env_mismatch','wrong_model'):
                    before=full_hash(router.capture(lane))
                    rejected(lambda:fork.install_at_selected_query(tick=0,device_uuid='CPU',model_sha256='b'*64 if mode=='wrong_model' else MODEL_SHA))
                    assert before==full_hash(router.capture(lane));checks+=2
                else:
                    fork.install_at_selected_query(tick=0,device_uuid='CPU',model_sha256=MODEL_SHA)
                    if mode=='duplicate_install':
                        rejected(lambda:fork.install_at_selected_query(tick=0,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                    else:
                        fork.finish_origin(22 if mode=='censor' else 32,end_tick=100,requested95_arrived=mode!='no95')
                        rejected(lambda:fork.draw_following(operation(m),query=1,tick=100,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
        assert not torch.cuda.is_initialized();checks+=1
    finally:router.uninstall()
    O=T/'outputs/native_nextchunk_v1'
    report=dict(passed=True,checks=checks,branches=records,
          source_sha256={str(C/n):sha(C/n) for n in ('sampler_current.py','current_sampler_checks.py','sampler_fork.py','sampler_checks.py')},
          source_method_AST_hashes=[h1,h2,h3],
          fixture_forward_only=True,NOT_GPU_READY=True,
          scope='Real pinned sample_noise/sample_actions/infer bodies and actual CPU RngRouter; exact selected/origin/next-query CPU RNG and nextbase versus independent sequential reference. Embedding/denoise/transforms are fixtures, not real VLA/physics/CUDA proof. Unlike preserved legacy-end handoff, correct native fork installs BEFORE origin and never uses anchorRGB for actor.')
    path=O/'current_sampler_cpu_checks.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print('CURRENT_NATIVE_SAMPLER_CPU_PASS',checks,'NO_REAL_VLA_CUDA_SIM',flush=True)
if __name__=='__main__':run_current()
