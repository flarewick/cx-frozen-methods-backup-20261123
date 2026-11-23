"""Original real cached gate inputs plus explicit veto/cooldown/budget controls."""
import copy,json
from types import SimpleNamespace
import numpy as np
from .contract import T,O,GATE_MODEL,read,mapped,sha
from .paired_heads import build,forward,ARM
from .gated_compose import EventComposer,FeedbackApplication

def main():
    import torch
    from cx_tta.multi_event import EventConfig,EventDetector,LocalController,apply_arm_candidate
    from cx_tta.empirical_onset_gate import candidate_packet
    from scripts.gate_all_candidate_v237.allgate237 import AllCandidateVeto
    torch.set_num_threads(1);before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):gate,aux,_,_=build()
    from candidate_eval_policy import scaled_actions
    b=T/'outputs/native_eval_throughput_v3/branch10';root=mapped(read(b/'execution_state.json')['root'])
    count=0;input_files=[]
    for case in sorted((root/'cases').iterdir()):
        composer=EventComposer(EventConfig());zero=EventComposer(EventConfig())
        detector=EventDetector(EventConfig());original=AllCandidateVeto(LocalController(EventConfig()),read(GATE_MODEL))
        for file in sorted((case/'M_event_full').glob('policy[0-9][0-9][0-9].npz')):
            with np.load(file,allow_pickle=False)as z:
                legal={k.removeprefix('legal_'):z[k].copy()for k in z.files if k.startswith('legal_')}
            base=legal['base'][0];reference=forward(gate,{k:v for k,v in legal.items()if k!='base'},base)['raw_delta'];q=composer.query
            packet=candidate_packet(legal,reference);event,evidence=detector.step(q,packet)
            decision=original.step(q,event,packet,None,enabled=True)
            _,raw=apply_arm_candidate(base,reference,decision['active']);expected=scaled_actions(base,raw,.5)
            aa,dd,mm=composer.step(base,reference,legal);zz,zd,zm=zero.step(base,reference,legal,adapted_ARM=reference[:,ARM].copy())
            assert np.array_equal(aa,expected[0])and np.array_equal(dd,expected[1])
            assert mm['decision']==decision and mm['event']==event and mm['evidence']==evidence
            assert np.array_equal(aa,zz)and np.array_equal(dd,zd)and mm['decision']==zm['decision']
            assert composer.controller.interventions==original.original.interventions
            count+=1;input_files.append(file)
    # Only the following tiny CPU fixtures change classifier coefficients.
    # Production always loads the SHA-verified frozen v118 JSON unchanged.
    model=read(GATE_MODEL)
    def fixture(admit):
        value=copy.deepcopy(model);value['raw_member_weights']=np.zeros((12,593)).tolist();value['raw_member_biases']=[100. if admit else -100.]*12
        ctl=LocalController(EventConfig());return SimpleNamespace(controller=ctl,gate=SimpleNamespace(original=ctl,model=value),history={0:dict(id=0,detected_query=0,kinds=['close_left'])},current=dict(query=1,legal=legal))
    def event(q):return dict(query=q,decision=dict(active=False,selected_event=None,interventions=0),adaptation_applied=False)
    def update(q):return dict(accepted=True,mode='TTA',source_query=0,query=q,fast_after='fixture_revision_'+str(q))
    reference=np.zeros((32,14),np.float32);direction=np.full((32,12),.001,np.float32);no_delta=np.zeros_like(base)
    reject=fixture(False);app=FeedbackApplication()
    a,d,e,receipt=app.step(reject,base,reference,direction,base.copy(),no_delta,event(1),update(1),condition='TTA')
    assert np.array_equal(a,base)and not np.any(d)and not receipt['applied']and reject.controller.interventions==0
    assert receipt['reason']=='frozen_v118_rejected_fast_difference'
    try:app.step(reject,base,reference,direction,base.copy(),no_delta,event(1),update(1),condition='TTA')
    except ValueError:pass
    else:raise AssertionError('Rejected revision reused')
    accept=fixture(True);app=FeedbackApplication()
    for q in range(1,8):
        accept.current['query']=q
        a,d,e,receipt=app.step(accept,base,reference,direction,base.copy(),no_delta,event(q),update(q),condition='TTA')
        assert np.array_equal(a[:,[6,13]],base[:,[6,13]])and np.max(np.abs(d))<=.000501
        assert receipt['applied']==(q in(1,3,5))
        assert accept.gate.original is accept.controller
    assert accept.controller.interventions==3
    zero=fixture(True);app=FeedbackApplication()
    a,d,e,receipt=app.step(zero,base,reference,np.zeros((32,12),np.float32),base.copy(),no_delta,event(1),update(1),condition='TTA')
    assert np.array_equal(a,base)and not np.any(d)and zero.controller.interventions==0
    assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
    value=dict(passed=True,actual_cached_native_queries=count,original_v118_all_event_controller_decisions_and_alpha05_actions_exact=True,zero_transferred_increment_actions_exact=True,explicit_fast_difference_veto_restores_original_actions_and_budget=True,zero_difference_no_action_or_budget=True,original_cooldown_and_max3_preserved=True,GRIP_base_exact=True,gate_fixture_never_used_for_GPU=True,CPU_RNG_preserved=True,CUDA_initialized=False,not_native_GPU_or_task_benefit_qualification=True,input_sha256={str(f):sha(f)for f in input_files},source_sha256={str(f):sha(f)for f in (T/'native_seven_task_gate_expansion_v1').glob('*.py')})
    output=O/'CPU_gate_composition_v2.json';assert not output.exists();output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items()if k not in('source_sha256','input_sha256')}))

if __name__=='__main__':main()
