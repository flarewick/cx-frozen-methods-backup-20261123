"""CPU acceptance of phase-target sources, exact forks, bounded targets and cache."""
import argparse
from pathlib import Path
import numpy as np
import torch,yaml
from .native_history_cache import read,arrays,require,verify
from .audit import digest
from .paths import write_json,output_path
from .structured_attribution_analysis import _load_trace,_probe
from .structured_action_adapter import FORWARD_KEYS,ARM_INDICES
from .output_support_adapter import OutputSupportAdapter,validate_native_arms
from .train_output_support import read_checkpoint
from .phase_target_policy import compose_grip,GRIPPERS

def summarize(run):
    run=Path(run);m=read(run/'manifest.json');p=read(m['protocol']);verify(p['source_sha256'])
    require(digest(Path(m['protocol']))==m['protocol_sha256'],'Frozen protocol')
    reg=read(Path(m['registry'])/'records.json');require(digest(Path(m['registry'])/'records.json')==m['registry_records_sha256'],'Frozen registry')
    es=read(run/'episodes.json');panels=read(run/'result.json')['panels']
    hw=read(run/'hardware.json');require(hw['compiled'] and hw['shared_sampler'] and not hw['global_dynamo_disabled'] and hw['sample_kwargs']=={},'Official common sampler')
    require(_probe(run)['passed'],'Exactzero originalL0')
    budgetfile=run/'robotwin/task_config/_eval_step_limit.yml';budget=yaml.safe_load(budgetfile.read_text())[m['task']]
    require(digest(budgetfile)==p['budget_config_sha256'][m['task']],'Original budget')
    payload=read_checkpoint(Path(p['training_run'])/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
    model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False);model.load_state_dict(payload['model'])
    traces={};packets={};errors=[];cpu_errors=[];queries=0;actions=0;sources={}
    for e in es:
        try:
            t=_load_trace(run,e);rows=t['rows'];numeric=t['arrays'];prefix=e.get('prefix_queries',0)
            require(len(rows)==e.get('probe_queries',e['queries']) and e['queries']==prefix+len(rows),'Trace/cache global queryclock')
            z=arrays(Path(e['cache'])/'queries.npz');labels=read(Path(e['cache'])/'offline_labels.json')['queries']
            require(set(z)==FORWARD_KEYS|{'base','raw_delta','applied_delta','actions','valid'} and len(z['base'])==e['queries']==len(labels),'Allqueryschema')
            goal=np.asarray(next(r for r in reg if r['task']==m['task'] and r['seed']==e['seed'])['expert_terminal_grippers'],np.float32)
            count=rows[0]['native_before']['control']['take_action_cnt'];first_event=None;window=None
            for i,row in enumerate(rows):
                q=prefix+i;require(row['query']==q and row['obs_updates']==32 and row['base_infer_calls']==row['auxiliary_prefix_calls']==1,'Original query/observations')
                require(row['rng_before_query']==row['rng_after_feature'] and row['rng_before_query']!=row['rng_after_sampler'],'Internalnoise/noninterference')
                require(set(row['structured_input_sha256'])==FORWARD_KEYS and row['has_history']==(q>0) and row['history_query']==(q-1 if q else None),'Nine legal realhistory')
                for k in FORWARD_KEYS:require(__import__('cx_tta.eval_diagnostics',fromlist=['array_hash']).array_hash(z[k][q:q+1])==row['structured_input_sha256'][k],'Actual saved9fields')
                for k in ('base','raw_delta','applied_delta'):require(z[k][q].tobytes()==numeric[k][i].tobytes(),'Numeric/cache consistency')
                b,raw=z['base'][q],z['raw_delta'][q]
                if e['condition'] in ('Z_zero','A_causal_arm'):
                    decrease=numeric['state'][i,[6,13]]-np.clip(b[:,[6,13]],0,1).min(0)
                    if first_event is None and np.any(decrease>=.1):first_event=q;window=[q+1,q+3]
                    active=e['condition']=='A_causal_arm' and window is not None and window[0]<=q<window[1]
                    want=np.zeros_like(b);act=b.copy()
                    if active:want[:,ARM_INDICES]=raw[:,ARM_INDICES];act[:,ARM_INDICES]=b[:,ARM_INDICES]+raw[:,ARM_INDICES]
                    require(row['window_active']==bool(active) and row['gate_decision']['window']==window,'Causal source mask')
                else:
                    active=q==e['point_query'];act,want=compose_grip(b,raw,e['condition'],active,goal if e['condition']=='T_stage_grip' else None)
                    require(row['teacher_privileged']==(e['condition']=='T_stage_grip') and row['probe_active']==bool(active),'Declared teacher/NNmask')
                    require(row['teacher_gripper_goal']==(goal.tolist() if e['condition']=='T_stage_grip' else None),'Sourceexpert goal notarbitrary')
                require(z['actions'][q].tobytes()==act.tobytes() and z['applied_delta'][q].tobytes()==want.tobytes(),'Actualboundedformula')
                from .eval_diagnostics import array_hash,json_hash
                require(row['action_chunk_sha256']==array_hash(act) and row['raw_delta_sha256']==array_hash(raw) and row['base_chunk_sha256']==array_hash(b),'Actual hashes')
                validate_native_arms(act,p['robot_joint_limits'])
                before,after=row['native_before']['control'],row['native_after']['control']
                require(before['step_lim']==after['step_lim']==budget and before['take_action_cnt']==count and after['take_action_cnt']==count+row['executed'],'Original nativeclock/budget')
                require(not before['eval_success'] and 0<row['executed']<=32,'No postterminalprogress')
                require(json_hash(row['native_before'])==row['native_before_sha256'] and json_hash(row['native_after'])==row['native_after_sha256'],'Readable physics hashes')
                require(np.array_equal(z['valid'][q],np.arange(32)<row['executed']),'Actualvalidtail,not observationcount')
                count=after['take_action_cnt'];queries+=1;actions+=row['executed']
                with torch.inference_mode():out=model({k:torch.from_numpy(z[k][q:q+1].copy()) for k in FORWARD_KEYS},torch.from_numpy(b[None].copy()),True)
                error=float(np.abs(out['delta'][0].numpy()-raw).max());require(error<2e-5,'ActualA reload');cpu_errors.append(error)
            require(count==e['executed_actions'] and count<=budget,'Terminal accounting')
            if e['complete']:require(type(e['success'])is bool and e['success']==after['eval_success'] and (e['success'] or count==budget),'Full terminal')
            else:require(e['success']is None and m['mode']=='smoke','Censored not counted')
            key=(e['seed'],e.get('source_condition',e['condition']),e['condition']);traces[key]=t;packets[key]=z
            for file in (Path(e['cache'])/'queries.npz',Path(e['cache'])/'manifest.json',run/'traces'/(e['trace_stem']+'.npz'),run/'traces'/(e['trace_stem']+'.jsonl.gz')):
                sources[str(file)]=digest(file)
        except (ValueError,OSError,KeyError,RuntimeError,AssertionError) as ex:errors.append(str((e['seed'],e['condition']))+': '+str(ex))
    qualified=[];forks=0
    for panel in panels:
        if panel.get('point_absent'):continue
        key=(panel['seed'],panel['source_condition']);src=traces.get((*key,key[1]))
        if src is None:errors.append('Missing source');continue
        point=read(run/'points'/src['episode']['trace_stem']/'point.json');q=point['query'];sz=packets[(*key,key[1])]
        require(np.count_nonzero(sz['applied_delta'][q:])==0,'Reused sourceCbase future must be exact base')
        source_row=src['rows'][q];children={}
        for case in ('A_grip','T_stage_grip'):
            t=traces.get((*key,case))
            if t is None:errors.append('Missing pairedchild');continue
            row=t['rows'][0];zz=packets[(*key,case)]
            for field in ('native_before_sha256','observation_sha256','structured_input_sha256','base_chunk_sha256','raw_delta_sha256','rng_before_query','rng_after_sampler'):
                require(row[field]==source_row[field]==point[field],'Exact commonphysical/observable/RNG/fork '+field)
            require(all(zz[k][:q].tobytes()==sz[k][:q].tobytes() for k in zz),'Original sourceprefix retained in cache')
            children[case]=t['episode'];forks+=1
        full=src['episode']['complete'] and len(children)==2 and all(e['complete'] for e in children.values())
        recover=bool(full and not src['episode']['success'] and children['T_stage_grip']['success'])
        require(panel['qualified_teacher_recovery']==recover,'No teacherqualificationfromsmoke')
        if m['mode']=='full':require(not point['forced_interface_only'] and point['source_state']['geometry_satisfied'] and not point['source_state']['native_success'],'Native stage selectedfromcurrentGTonlyoffline')
        if recover:qualified.append(dict(seed=panel['seed'],source_condition=panel['source_condition'],query=q,teacher=True,adapter_improvement=False))
    require(len(traces)==len(es),'Every branch readback')
    return dict(status='passed' if not errors else 'failed',interface_passed=not errors,mode=m['mode'],errors=errors,
        actual_postfork_or_source_queries_checked=queries,actual_actions_checked_excludingreplayedprefix=actions,actual_A_CPU_max_error=max(cpu_errors,default=None),
        paired_forks_checked=forks,qualified_teacher_recoveries=qualified,distinct_qualified_scenes=len({r['seed'] for r in qualified}),
        records=len(es),complete_rollouts=sum(e['complete'] for e in es),censored=sum(not e['complete'] for e in es),
        training_approved=False,teacher_success_not_adapter_improvement=True,source_sha256=sources,GPU_started=False)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();torch.set_num_threads(2)
    out=summarize(output_path(a.run));write_json(a.run/'CPU_audit.json',out);require(out['interface_passed'],'Phase-target acceptance');print('PHASE_CPU_PASS',out['records'],out['paired_forks_checked'],out['distinct_qualified_scenes'],flush=True)
if __name__=='__main__':main()
