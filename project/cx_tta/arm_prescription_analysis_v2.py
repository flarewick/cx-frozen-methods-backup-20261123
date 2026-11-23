"""V2 readback: distinguish raw float64, native float32, and ARM/pi model state."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml

from .audit import digest
from .paths import output_path, write_json
from .native_history_cache import read, arrays, require, verify
from .structured_attribution_analysis import _load_trace, _probe
from .structured_action_adapter import FORWARD_KEYS, ARM_INDICES
from .output_support_adapter import OutputSupportAdapter, validate_native_arms
from .train_output_support import read_checkpoint
from .eval_diagnostics import array_hash, json_hash
from .deploy import CAMERA_KEYS, POLICY_CAMERA_KEYS
from .causal_arm_analysis import order
from .local_benefit_cache import load_observables

CONDITIONS = ('Z_zero', 'F_first_full', 'B_two_full', 'H_two_half')


def summarize(run):
    run = Path(run)
    m = read(run/'manifest.json'); p = read(m['protocol'])
    require(p['stage'] == m['stage'] == 'local_arm_prescription_TRAIN', 'Registered stage')
    require(digest(Path(m['protocol'])) == m['protocol_sha256'], 'Protocol identity')
    verify(p['source_sha256'])
    reg = read(Path(m['registry'])/'records.json')
    require(digest(Path(m['registry'])/'records.json') == m['registry_records_sha256'], 'Registry identity')
    seeds = [s for s in p['seeds'] if next(r for r in reg if r['task']==m['task'] and r['seed']==s)['eligible']]
    if m['mode'] == 'smoke': seeds = seeds[:1]
    require(m['seeds'] == seeds and m['conditions'] == list(CONDITIONS), 'No outcome-based scene/condition changes')
    expected = [dict(seed=s, repeat=b, condition=c) for s,b,c in order(seeds, CONDITIONS, m['block_id'])]
    episodes = read(run/'episodes.json')
    require(expected == m['order'] == [{k:e[k] for k in ('seed','repeat','condition')} for e in episodes], 'Complete scheduled panel')
    hw = read(run/'hardware.json')
    require(hw['compiled'] and hw['shared_sampler'] and not hw['global_dynamo_disabled'] and hw['sample_kwargs']=={}, 'Official common sampler')
    require(_probe(run)['passed'], 'Zero equals original L0')
    bf = run/'robotwin/task_config/_eval_step_limit.yml'
    require(digest(bf) == p['budget_config_sha256'][m['task']], 'Original budget file')
    budget = yaml.safe_load(bf.read_text())[m['task']]
    payload = read_checkpoint(Path(p['training_run'])/'A_bounded_clipped_seed0.pt', 'A_bounded_clipped')
    model = OutputSupportAdapter(32, payload['metadata']['geometry_mean'], 'A_bounded_clipped').eval().requires_grad_(False)
    model.load_state_dict(payload['model'])
    traces = {}; errors = []; cpu_errors = []; sources = {}; events = []
    total_actions = active_actions = active_queries = total_queries = 0
    for e in episodes:
        key = (e['seed'], e['condition'])
        try:
            t = _load_trace(run, e); rows, numeric = t['rows'], t['arrays']
            require(len(rows) == e['queries'] and bool(rows), 'Actual query count')
            folder = Path(e['cache']); x, base = load_observables(folder); z = arrays(folder/'queries.npz')
            cm = read(folder/'manifest.json'); labels = read(folder/'offline_labels.json')['queries']
            require(cm['queries'] == len(rows) == len(labels) and cm['episode_success'] == e['success'] and cm['training_approved'] is False, 'Actual cache completion/label isolation')
            rr = next(r for r in reg if r['task']==m['task'] and r['seed']==e['seed'])
            require(rows[0]['native_before_sha256']==rr['initial_physical_sha256']==e['initial_native_sha256'], 'Registered initial physics')
            comps = rows[0]['observation_components']; sig = rr['initial_observation_signature']
            ctrl = rows[0]['native_before']['control']
            registered_command = np.asarray(ctrl['left_command'] + ctrl['right_command'], np.float64)
            require(registered_command.shape == (14,) and array_hash(registered_command)==sig['command_state']
                    and array_hash(registered_command.astype(np.float32))==comps['command_state']
                    and all(comps[b]==sig['images'][a] for a,b in zip(CAMERA_KEYS,POLICY_CAMERA_KEYS)),
                    'Registered raw float64 command, model float32 command, and initial images')
            first = None; window = None; count = 0; actual_on = 0
            for q, row in enumerate(rows):
                b, raw, d = (z[k][q] for k in ('base','raw_delta','applied_delta'))
                require(row['query']==q and row['evaluated_condition']==e['condition'] and row['derived_checkpoint_sha256']==p['checkpoint_sha256'], 'Actual query/model identity')
                require(row['has_history']==(q>0) and row['history_query']==(q-1 if q else None) and set(row['structured_input_sha256'])==FORWARD_KEYS, 'Only actual legal history')
                for k in FORWARD_KEYS:
                    require(array_hash(x[k][q:q+1])==row['structured_input_sha256'][k], 'Actual input hash '+k)
                for k in ('visual','text','state','ee_pose'):
                    want=x[k][q-1:q] if q else np.zeros_like(x['past_'+k][q:q+1])
                    require(x['past_'+k][q:q+1].tobytes()==want.tobytes(), 'Actual preceding input '+k)
                scaled_state = numeric['state'][q].copy()
                scaled_state[list(ARM_INDICES)] /= np.pi
                require(x['state'][q].tobytes()==scaled_state.tobytes()
                        and float(x['has_history'][q,0])==float(q>0), 'Actual ARM/pi state and history flag')
                for k in ('base','raw_delta','applied_delta'):
                    require(z[k][q].tobytes()==numeric[k][q].tobytes(), 'Stored numeric/cache consistency')
                require(row['base_infer_calls']==row['auxiliary_prefix_calls']==1 and row['obs_updates']==32, 'One sample/per-action observation')
                require(row['rng_before_query']==row['rng_after_feature'] and row['rng_before_query']!=row['rng_after_sampler'], 'Internal noise/noninterference')
                dec=numeric['state'][q,[6,13]]-np.clip(b[:,[6,13]],0,1).min(0)
                event=bool(np.any(dec>=.1)); new=first is None and event
                if new: first=q; window=[q+1,q+3]
                expected_decision=dict(query=q, closing_decrease=dec.tolist(), event_now=event,new_event=new,first_event_query=first,
                    window=window,active=window is not None and window[0]<=q<window[1],
                    state_sha256=array_hash(numeric['state'][q]),base_sha256=array_hash(b),threshold=.1)
                require(row['gate_decision']==expected_decision, 'Causal event replay')
                duration = {'Z_zero':0,'F_first_full':1,'B_two_full':2,'H_two_half':2}[e['condition']]
                chosen = None if window is None else [window[0],window[0]+duration]
                on = chosen is not None and chosen[0]<=q<chosen[1]
                strength = (.5 if e['condition']=='H_two_half' else 1.) if on else 0.
                want=np.zeros_like(b); act=b.copy()
                if on: want[:,ARM_INDICES]=raw[:,ARM_INDICES]*np.float32(strength)
                changed=want!=0; act[changed]=b[changed]+want[changed]
                require(np.max(np.abs(raw)-np.asarray(p['caps']))<1e-6, 'Original candidate bounds')
                require(row['prescription']==dict(window=chosen,active=on,strength=strength) and row['arm_strength']==strength and row['gripper_strength']==0., 'Local prescription replay')
                require(d.tobytes()==want.tobytes() and z['actions'][q].tobytes()==act.tobytes(), 'Actual action formula')
                for k,value in (('base_chunk_sha256',b),('raw_delta_sha256',raw),('applied_delta_sha256',d),('action_chunk_sha256',act)):
                    require(row[k]==array_hash(value), 'Recorded command hash')
                require(row['robot_domain']==validate_native_arms(act,p['robot_joint_limits']), 'Original domain')
                require(act[:,[6,13]].tobytes()==b[:,[6,13]].tobytes(), 'Grippers exactly base')
                before,after = row['native_before']['control'],row['native_after']['control']
                require(all(json_hash(row['native_'+s])==row['native_'+s+'_sha256'] for s in ('before','after')), 'Physical snapshot hashes')
                require(before['step_lim']==after['step_lim']==budget and not before['eval_success'] and before['take_action_cnt']==count, 'Original budget/start count')
                require(0<row['executed']<=32 and after['take_action_cnt']==count+row['executed']<=budget, 'Actual native progress')
                require(z['valid'].dtype==np.bool_ and np.array_equal(z['valid'][q],np.arange(32)<row['executed']), 'Actual valid prefix')
                count=after['take_action_cnt']; total_actions+=row['executed']; total_queries+=1
                if on: active_actions+=row['executed']; active_queries+=1; actual_on+=row['executed']
                with torch.inference_mode():
                    out=model({k:torch.from_numpy(v[q:q+1].copy()) for k,v in x.items()},torch.from_numpy(base[q:q+1].copy()),True)
                error=float(np.abs(out['delta'][0].numpy()-raw).max())
                require(error<2e-5, 'Frozen A numerical reload'); cpu_errors.append(error)
            require(count==e['executed_actions'], 'Terminal action count')
            if e['complete']:
                require(m['mode']=='full' and type(e['success']) is bool and e['success']==after['eval_success'] and (e['success'] or count==budget), 'Complete original-budget outcome')
            else:
                require(m['mode']=='smoke' and e['success'] is None, 'Censored outcome not counted')
            events.append(dict(seed=e['seed'],condition=e['condition'],first_event=first,opportunity_window=window,
                corrected_actions=actual_on,no_trigger=first is None,opportunity_reached=window is not None and len(rows)>window[0]))
            traces[key]=t
            for file in (folder/'queries.npz',folder/'manifest.json',folder/'offline_labels.json',
                         run/'traces'/(e['trace_stem']+'.npz'),run/'traces'/(e['trace_stem']+'.jsonl.gz')):
                sources[str(file)]=digest(file)
        except (ValueError,KeyError,OSError,RuntimeError,AssertionError) as exc:
            errors.append(str(key)+': '+str(exc))
    require(len(traces)==len(episodes), 'Every episode accepted: '+str(errors))
    pairs=[]; decision_labels=[]
    fields=('native_before_sha256','observation_sha256','structured_input_sha256',
            'rng_before_query','rng_after_sampler','base_chunk_sha256','raw_delta_sha256')
    for seed in seeds:
        combinations=[('Z_zero',c,'onset') for c in CONDITIONS if c!='Z_zero']
        combinations.append(('F_first_full','B_two_full','continue_after_first'))
        for ca,cb,kind in combinations:
            a,b=traces[(seed,ca)],traces[(seed,cb)]
            w=b['rows'][-1]['gate_decision']['window']
            q=None if w is None else w[0]+int(kind=='continue_after_first')
            shared=min(len(a['rows']),len(b['rows']),q+1 if q is not None else max(len(a['rows']),len(b['rows'])))
            for i in range(shared):
                require(all(a['rows'][i][k]==b['rows'][i][k] for k in fields), 'Exact paired pre-query '+str((seed,ca,cb,i)))
                if q is None or i<q:
                    require(all(a['rows'][i][k]==b['rows'][i][k] for k in ('native_after_sha256','action_chunk_sha256')), 'Identical executed shared prefix')
            reached=q is not None and q<len(a['rows']) and q<len(b['rows'])
            full=a['episode']['complete'] and b['episode']['complete']
            pairs.append(dict(seed=seed,a=ca,b=cb,kind=kind,passed=True,query=q,shared_queries=shared,decision_reached=reached))
            if reached:
                decision_labels.append(dict(seed=seed,task=m['task'],block=m['block_id'],kind=kind,query=q,
                    input_cache=a['episode']['cache'],input_sha256=a['rows'][q]['structured_input_sha256'],
                    control=ca,candidate=cb,control_success=a['episode']['success'],candidate_success=b['episode']['success'],
                    complete=full,incremental_success=int(b['episode']['success'])-int(a['episode']['success']) if full else None,
                    label_scope='This shared-state finite prescription only; not dense query labels or oracle online inputs',training_approved=False))
    outcomes={};changes={}
    for c in CONDITIONS:
        es=[e for e in episodes if e['condition']==c]; full=[e for e in es if e['complete']]
        outcomes[c]=dict(successes=sum(e['success'] for e in full),completed=len(full),censored=len(es)-len(full),
            ratio=sum(e['success'] for e in full)/len(full) if full else None)
        if c!='Z_zero':
            labels=Counter()
            for seed in seeds:
                a,b=traces[(seed,'Z_zero')]['episode'],traces[(seed,c)]['episode']
                labels['censored' if not(a['complete'] and b['complete']) else 'tie' if a['success']==b['success'] else 'rescued' if b['success'] else 'harmed']+=1
            changes[c]={k:labels[k] for k in ('rescued','harmed','tie','censored')}
            changes[c]['net']=labels['rescued']-labels['harmed']
    for name in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz','hardware.json'):
        sources[str(run/name)]=digest(run/name)
    return dict(status='passed',interface_passed=not errors,stage=p['stage'],mode=m['mode'],task=m['task'],
        block_id=m['block_id'],process_instance_id=m['process_instance_id'],errors=errors,
        records=len(episodes),distinct_scenes=len(seeds),complete_rollouts=sum(e['complete'] for e in episodes),
        censored=sum(not e['complete'] for e in episodes),actual_queries=total_queries,actual_actions=total_actions,
        active_queries=active_queries,corrected_actions=active_actions,CPU_proposal_max_error=max(cpu_errors),
        outcomes=outcomes,changes=changes,events=events,paired_decisions=pairs,decision_labels=decision_labels,
        training_approved=False,GT_used_in_actor=False,source_sha256=sources,
        audit_version=2,auditor_sha256={str(Path(__file__).resolve()):digest(Path(__file__))})


def main():
    ap=argparse.ArgumentParser(__doc__); ap.add_argument('run',type=Path); a=ap.parse_args()
    torch.set_num_threads(2)
    run=output_path(a.run); result=summarize(run); write_json(run/'CPU_audit.json',result)
    print('ARM_PRESCRIPTION_CPU_PASS',result['records'],result['actual_queries'],result['actual_actions'],flush=True)


if __name__=='__main__': main()
