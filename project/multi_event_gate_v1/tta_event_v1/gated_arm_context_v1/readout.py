"""Read-only scoring diagnosis, not a trajectory or success counterfactual."""
import json
import numpy as np
from native_seven_task_gate_expansion_v1.contract import T,GATE_MODEL,read,mapped

def main():
    import sys
    from native_seven_task_gate_expansion_v1.contract import W
    sys.path.insert(0,str(W/'task_adapter_v1'))
    from cx_tta.empirical_onset_gate import candidate_packet
    from native_arm_phase_feedback_v1.readouts import KIND
    from benefit_fixed_ensemble_gate_v119 import predict
    b=T/'outputs/native_seven_gated_dev_v2/dev_gated';root=mapped(read(b/'execution_state.json')['root']);model=read(GATE_MODEL);rows=[]
    for file in sorted(root.glob('cases/*/TTA/result.json')):
        result=read(file);queries=result['queries']
        for query in queries:
            receipt=query['feedback_application_receipt']
            if receipt.get('reason')!='missing_causal_gate_context':continue
            source=receipt['source_query'];admission=queries[source]['feedback_source_admission']
            assert source==0 and admission['admitted']and admission['kind']==KIND and admission['reason']=='initial_actual_own_ARM_commands'
            with np.load(mapped(query['policy']['policy_file']),allow_pickle=False)as z:legal={k.removeprefix('legal_'):z[k].copy()for k in z.files if k.startswith('legal_')}
            with np.load(mapped(query['actions_file']),allow_pickle=False)as z:correction=(z['gate_candidate']-z['reference']).astype(np.float32)
            active=[r['query']for r in queries[:query['query']]if r['event']['decision']['active']]
            last=active[-1]if active else None
            context=np.asarray([0.]*5+[min(query['query']-source,8)/8,min(query['event']['decision']['interventions'],8)/8,1. if last is None else min(query['query']-last,8)/8],np.float32)
            scores=predict(model,candidate_packet(legal,correction),context)
            rows.append(dict(case=file.parent.parent.name,query=query['query'],source_kind=KIND,actual_initial_ARM_source=True,no_fabricated_gripper_or_progress_event=True,context=context.tolist(),scores=scores,admit=scores['rescue']-2*scores['harm']>0,maximum_raw_fast_difference=float(np.max(np.abs(correction))),not_executed_not_task_benefit=True))
    out=T/'outputs/gated_arm_context_v1';out.mkdir(exist_ok=True);file=out/'CPU_original_prefix_score_diagnosis.json';assert not file.exists();file.write_text(json.dumps(dict(passed=True,rows=rows,scope='Only observed-prefix feature/score diagnosis; no updated execution or success claim'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
