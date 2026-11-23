"""Frozen S0 pre-tanh inspection on existing legal native-entry inputs.

Linear reconstruction is a post-hoc diagnostic, never evidence that an old
clipped-label model learned raw teacher amplitudes or improves task success.
"""
from pathlib import Path
import json

import numpy as np
import torch

from .audit import digest
from .native_history_cache import ROOT,BASE,arrays,read,verify,require
from .paths import new_run,write_json
from .structured_action_adapter import StructuredActionAdapter,FORWARD_KEYS,ARM_INDICES
from .structured_action_deploy import read_checkpoint

CACHE=ROOT/'outputs/feature_caches/native_history_20261005_030329_014198'
ACCEPTED=ROOT/'outputs/reports/protocols/native_entry_bounded_v2_20261005_045526_497512/accepted_results.json'


def alignment(predicted,target):
    a=predicted.astype(float).ravel();b=target.astype(float).ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den<1e-15 else float(np.dot(a,b)/den)


def main():
    accepted=read(ACCEPTED);verify(accepted['source_sha256']);cp=read(CACHE/'protocol.json')
    verify(cp['source_sha256']);require(read(CACHE/'audit.json')['passed'],'Native cache audit required')
    payload,_=read_checkpoint(BASE/'S0_seed0.pt','S0');meta=payload['metadata']
    model=StructuredActionAdapter(meta['channels'],meta['geometry_mean']).eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True)
    x=arrays(CACHE/'inputs.npz');require(set(x)==FORWARD_KEYS,'Exactly9 legal fields')
    caps=model.caps.numpy();rows=[];logits=[]
    hook=model.action_head.register_forward_hook(lambda module,args,out:logits.append(out.detach().clone()))
    sources={str(ACCEPTED):digest(ACCEPTED),str(CACHE/'protocol.json'):digest(CACHE/'protocol.json'),
        str(CACHE/'inputs.npz'):digest(CACHE/'inputs.npz'),str(BASE/'S0_seed0.pt'):digest(BASE/'S0_seed0.pt'),
        str(Path(__file__)):digest(Path(__file__))}
    try:
        for key in ('local_shoe_run','reused_external_switch_run'):
            run=Path(accepted[key]);m=read(run/'manifest.json');point=m['point'];index=point['cache_row'];row=cp['rows'][index]
            require(row['signature']==point['source_signature'] and row['task']==point['task']
                and row['seed']==point['seed'] and row['native_count']==point['native_count'],'Cached fork input differs')
            y=arrays(run/'first_live_base_target.npz');base=y['base'];valid=y['valid'];target=y['target']
            inputs={k:torch.from_numpy(v[index:index+1].copy()) for k,v in x.items()}
            logits.clear()
            with torch.inference_mode():output=model(inputs,torch.from_numpy(base[None].copy()))
            require(len(logits)==1,'One actual head call required')
            z=logits[0][0].numpy();bounded=output['delta'][0].numpy();linear=z*caps
            np.testing.assert_allclose(bounded,np.tanh(z)*caps,rtol=2e-6,atol=1e-8)
            teacher=target-base;arm=np.array(ARM_INDICES)
            arm_b=bounded[valid][:,arm];arm_l=linear[valid][:,arm];arm_t=teacher[valid][:,arm]
            critical=(np.abs(teacher)>caps+2e-6)&valid[:,None];critical[:,[6,13]]=False
            same_sign=(bounded[critical]*teacher[critical]>0)
            metrics=dict(task=point['task'],seed=point['seed'],native_count=point['native_count'],
                cached_legal_input_row=index,same_actual_live_base=True,head_calls=1,
                teacher_arm_delta_max_rad=float(np.abs(arm_t).max()),
                predicted_bounded_arm_delta_max_rad=float(np.abs(arm_b).max()),
                reconstructed_linear_arm_delta_max_rad=float(np.abs(arm_l).max()),
                near_cap_arm_fraction=float((np.abs(arm_b/caps[arm])>=.95).mean()),
                linear_over_original_cap_arm_fraction=float((np.abs(arm_l)>caps[arm]+2e-6).mean()),
                bounded_arm_cosine_to_teacher=alignment(arm_b,arm_t),
                reconstructed_linear_arm_cosine_to_teacher=alignment(arm_l,arm_t),
                teacher_over_cap_arm_coordinates=int(critical.sum()),
                predicted_sign_agreement_on_over_cap_coordinates=float(same_sign.mean()) if same_sign.size else None,
                activation_removal_keeps_signs=bool(np.array_equal(np.sign(z),np.sign(np.tanh(z)))),
                training_label_was_clipped=True,linear_output_was_not_trained=True,
                physical_effect_measured=False,training_or_deployment_allowed_by_this_check=False)
            rows.append(metrics)
            sources[str(run/'first_live_base_target.npz')]=digest(run/'first_live_base_target.npz')
    finally:hook.remove()
    history=[]
    for name in ('pose_capacity_20261002_045559_185861','coverage_20261002_092902_991797','scene_crossfit_20261002_114334_518712'):
        d=ROOT/'outputs/effect_learning'/name;p=read(d/'protocol.json');r=read(d/'result.json')
        history.append(dict(run=str(d),status=r['status'],scope=p['scope'],evidence='Offline learned proposal; not a task-success evaluation'))
        for f in (d/'protocol.json',d/'result.json'):sources[str(f)]=digest(f)
    for f in (ROOT/'cx_tta/structured_action_adapter.py',ROOT/'cx_tta/pose_adapter.py',
        BASE/'training_protocol.json'):sources[str(f)]=digest(f)
    out=new_run('output_range_audit',group=('reports','attribution'))
    write_json(out/'result.json',dict(status='complete_CPU_inspection',rows=rows,historical_unbounded_learners=history,
        current_S2j_S2l_S2m_have_capped_outputs_and_clipped_labels=True,
        existing_teacher_raw_full_rollouts_are_not_learned_uncapped_head=True,
        current43k_output_only_vs_label_only_uncapping_closed_loop_found=False,
        model_parameters=sum(p.numel() for p in model.parameters()),GPU_started=False,
        trained_model_modified=False,source_sha256=sources,
        scope='Two existing selected TRAIN states, original final S0 with cached legal native history and actual live base. '
            'Pre-activation arithmetic only, not population, success, new training or hidden-state saturation causality.'))
    print('REPORT',out/'result.json',flush=True)
    for row in rows:print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
