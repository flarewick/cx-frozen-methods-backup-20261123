"""S2o same-process L0 and four matched learned heads, no video/RGB."""
import argparse,atexit,gzip,importlib,json,os,signal,time,traceback
from pathlib import Path
from .audit import digest
from .paths import ROOT,new_run,output_path,write_json,replace_json
from .native_history_cache import read,verify
from .train_output_support import BASE,DESIGN,CONDITIONS as CELLS,read_checkpoint
from .evaluate_structured_attribution import CompactTrace,PhysicalPolicy,pilot_reference,steady_probe
from .output_support_evaluation_analysis import CONDITIONS,InitialPairingGuard,evaluation_order,summarize


def prepare_protocol(run):
    q=read(run/'training_protocol.json');design=read(DESIGN);sources=dict(q['source_sha256']);references={}
    for task in design['evaluation_design']['tasks']:
        _,records,_,evidence=pilot_reference(task,BASE)
        for seed in design['evaluation_design']['scene_seeds'][task]:
            if not records[seed]['eligible']:raise ValueError('Ineligible fixed scene')
        references[task]=evidence
    checkpoints={}
    for c in CELLS:
        f=run/(c+'_seed0.pt');read_checkpoint(f,c);checkpoints[c]=digest(f)
    checkpoints['L0']=checkpoints['A_bounded_clipped']
    paths=[run/f for f in ('training_result.json','training_audit.json','control_reproduction.json')]
    paths += [ROOT/f for f in ('cx_tta/output_support_policy.py','cx_tta/output_support_evaluation_analysis.py',
        'cx_tta/evaluate_output_support.py','scripts/eval_output_support.sh')]
    sources.update({str(f):digest(f) for f in paths})
    write_json(run/'evaluation_protocol.json',dict(stage='output_support',training_run=str(run),conditions=list(CONDITIONS),
        checkpoints=checkpoints,seeds=design['evaluation_design']['scene_seeds'],repeats=2,reference=references,
        source_sha256=sources,reference_scales=q['reference_scales_native_units'],robot_joint_limits=q['robot_joint_limits'],
        robot_domain=q['robot_domain'],compiled=True,internal_noise=True,per_action_observations=True,
        shared_base_models=1,shared_samplers=1,native_budget=True,no_video=True,
        warmup='2 zero-correction warmups,4 restored internal-RNG zero probes; recreate scenes',
        scope='Five same-process conditions; fixed reused8 development scenes, not blind confirmation'))
    print('EVALUATION_PROTOCOL',run/'evaluation_protocol.json',flush=True)


def validate_protocol(run,task):
    p=read(run/'evaluation_protocol.json');verify(p['source_sha256']);design=read(DESIGN)
    if p['stage']!='output_support' or p['conditions']!=list(CONDITIONS) or p['seeds']!=design['evaluation_design']['scene_seeds']:
        raise ValueError('Sealed evaluator contract')
    for c in CELLS:
        path=run/(c+'_seed0.pt');read_checkpoint(path,c)
        if digest(path)!=p['checkpoints'][c]:raise ValueError('Checkpoint changed')
    _,reference,_,evidence=pilot_reference(task,BASE)
    return p,reference,evidence

class PairedTrace(CompactTrace):
    def __init__(self,run,stem,physical,guard,seed,repeat,condition):
        super().__init__(run,stem,physical)
        self.guard,self.context=guard,(seed,repeat,condition)
        self.legal_path=run/'legal_inputs'/(stem+'.npz')
        self.legal_path.parent.mkdir(exist_ok=True)
        self.legal_rows=[]

    def record(self,**kwargs):
        first=self.first is None
        super().record(**kwargs)
        if first:self.guard.observe(*self.context,self.first,self.numeric['base'][0])
        actual=self.policy.policy.last_legal_inputs
        if actual is not None:
            self.legal_rows.append({key:value.copy() for key,value in actual.items()})
            if len(self.legal_rows)>2:raise RuntimeError('First2 input capture exceeded')
            import numpy as np
            temporary=self.legal_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**{key:np.stack([r[key] for r in self.legal_rows]) for key in actual})
            temporary.replace(self.legal_path)


def execute(args,storage):
    from .train_output_support import BASE as BASE_RUN
    TRAIN_CONDITIONS=list(CELLS)
    from .evaluate_b2 import configure_inference, resolve_tokenizer, run_episode
    from .evaluate_structured_action import check_training_policy_identity, SmokeBudgetReached
    protocol,reference,ref_evidence=validate_protocol(args.training_run,args.task)
    tokenizer=resolve_tokenizer(args.tokenizer_path)
    identity=check_training_policy_identity(BASE_RUN,tokenizer)
    seeds=protocol['seeds'][args.task];order=evaluation_order(seeds,smoke=args.max_chunks is not None)
    candidate=args.training_run/'A_bounded_clipped_seed0.pt'
    manifest=dict(schema_version=1,stage='output_support',task=args.task,
        training_run=str(args.training_run),original_training_run=str(BASE_RUN),
        evaluation_protocol=str(args.training_run/'evaluation_protocol.json'),
        evaluation_protocol_sha256=digest(args.training_run/'evaluation_protocol.json'),
        seeds=seeds,conditions=list(CONDITIONS),repeats=2 if args.max_chunks is None else 1,
        order=[dict(seed=s,repeat=r,condition=c) for s,r,c in order],max_chunks=args.max_chunks,
        checkpoint_sha256=protocol['checkpoints'],compiled=True,policy_init_seed=0,
        rollout_protocol='official',noise_source='model_internal_torch',task_config='demo_randomized',
        instruction_type='unseen',no_video=True,original_native_budget=True,scene_timeout=args.scene_timeout,
        shared_base_models=1,shared_compiled_samplers=1,reference=ref_evidence,acceptance='Independent four-head CPU audit and exact A control reproduction',
        warmup=protocol['warmup'],scope=protocol['scope'])
    manifest['legal_input_capture']=dict(first_queries=2,fields='nine legal fields+base',task_budget_bytes=524288)
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest,gpu_started=False,requested_rollouts=len(order)),ensure_ascii=False))
        return
    import numpy as np
    import torch
    from .cache_expert import create_base_policy
    from .collect_smoke import make_config
    from .output_support_policy import OutputSupportPolicy
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU visible; no model or scene started')
    run=new_run('paired_smoke' if args.max_chunks else 'paired',group=('diagnostics','output_support',args.task))
    print('RUN',run,flush=True);write_json(run/'manifest.json',manifest)
    write_json(run/'training_identity_audit.json',identity)
    write_json(run/'storage_note.json',dict(images=False,video=False,model_copies=False,
        traces='All executed queries: four compact float32 arrays and compressed passive state metadata',
        legal_inputs='First2 queries per rollout, nine actual model fields and base only, <=256KiB/task',
        compiler='Private /tmp scratch, removed after worker teardown; static resources shared read-only'))
    episodes=[];started=time.monotonic();guard=InitialPairingGuard()
    try:
        storage.runtime_source=Path(ref_evidence['run'])/'robotwin'
        runtime=storage.prepare_runtime(run);storage.configure_runtime(run,runtime,tokenizer)
        args.task_config,args.instruction_type='demo_randomized','unseen'
        config=make_config(runtime,args,run)
        config.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
        write_json(run/'resolved_config.json',config)
        old_provenance=read(Path(ref_evidence['run'])/'provenance.json')
        runtime_sha={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*.py') if '__pycache__' not in p.parts}
        if runtime_sha!=old_provenance['runtime_sha256']:raise ValueError('Native runtime differs from sealed reference')
        for path,expected in old_provenance['code_sha256'].items():
            if digest(Path(path))!=expected:raise ValueError('Frozen source changed: '+path)
        code={str(p):digest(p) for p in (ROOT/'cx_tta').glob('*.py')}
        write_json(run/'provenance.json',dict(runtime_sha256=runtime_sha,source_sha256=code,
            identity=identity['live_identity'],policy_source_sha256=identity['policy_source_sha256']))
        task=getattr(importlib.import_module('envs.'+args.task),args.task)()
        configure_inference('compiled',0);torch.set_num_threads(2)
        replace_json(run/'progress.json',dict(phase='loading_one_shared_base',completed=0,requested=len(order)))
        base=create_base_policy();base._model.eval().requires_grad_(False)
        wrappers={c:OutputSupportPolicy(base,args.training_run,c) for c in CONDITIONS}
        wrappers['S0_zero']=OutputSupportPolicy(base,args.training_run,'A_bounded_clipped',force_zero=True)
        for wrapper in wrappers.values():
            wrapper.check_robot_config(config)
            if wrapper.sampler is not base._sample_actions:raise RuntimeError('Sampler object identity changed')
        write_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            compiled=hasattr(base._sample_actions,'_torchdynamo_orig_callable'),global_dynamo_disabled=bool(torch._dynamo.config.disable),
            shared_sampler=True,sampler_process=os.getpid(),matmul_precision=torch.get_float32_matmul_precision(),sample_kwargs=base._sample_kwargs))
        def timeout(*_):raise TimeoutError('Declared scene/probe timeout exceeded')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(args.scene_timeout)
        replace_json(run/'progress.json',dict(phase='common_warmup_and_fixed_input_probe',completed=0,requested=len(order)))
        probe=steady_probe(task,config,seeds[0],reference[seeds[0]]['instruction'],{c:wrappers[c] for c in ('L0','S0_zero')})
        np.savez_compressed(run/'steady_probe.npz',base=probe.pop('base_values'));write_json(run/'steady_probe.json',probe)
        signal.alarm(0)
        if not probe['passed']:raise RuntimeError('New-head-zero fixed-input probe failed before rollouts')
        for seed,repeat,condition in order:
            signal.alarm(args.scene_timeout)
            replace_json(run/'progress.json',dict(phase='output_support_factorial',completed=len(episodes),requested=len(order),seed=seed,repeat=repeat,condition=condition))
            wrapper=wrappers[condition];wrapper.reset_history();wrapper.episode_seed=seed
            physical=PhysicalPolicy(wrapper,task,args.max_chunks)
            stem=f'seed{seed}_r{repeat}_{condition}'
            trace=PairedTrace(run,stem,physical,guard,seed,repeat,condition)
            try:
                row=run_episode(task,config,runtime,seed,seeds.index(seed),physical,condition!='L0',
                    instruction_type='unseen',instruction_override=reference[seed]['instruction'],trace=trace,rollout_protocol='official')
                row['complete']=True
                if args.max_chunks:
                    row['observed_native_success']=row.pop('success');row.update(success=None,complete=False,reason='interface_smoke_no_rate_claim')
            except SmokeBudgetReached as error:row=error.record
            except Exception as error:
                row=dict(seed=seed,eligible=True,complete=False,success=None,reason='error',error_type=type(error).__name__,traceback=traceback.format_exc())
            row.update(repeat=repeat,condition=condition,queries=wrapper.query_count,trace_stem=stem)
            if trace.first:row.update(initial_native_sha256=trace.first['native_before_sha256'],initial_observation_sha256=trace.first['observation_sha256'])
            episodes.append(row);replace_json(run/'episodes.json',episodes);replace_json(run/'initial_pairing.json',guard.summary(order))
            print(json.dumps(row,ensure_ascii=False),flush=True);signal.alarm(0)
            if row.get('reason')=='error':break
        result=summarize(run);result.update(elapsed_seconds=time.monotonic()-started,
            numeric_bytes=sum(p.stat().st_size for p in (run/'traces').glob('*.npz')),
            trace_bytes=sum(p.stat().st_size for p in (run/'traces').glob('*.jsonl.gz')))
        write_json(run/'result.json',result)
        write_json(run/'source_runtime_audit.json',storage.verify_source());replace_json(run/'progress.json',dict(phase='finished',gate_passed=result['gate_passed'],interface_passed=result['interface_passed']))
        for condition in CONDITIONS:
            record=result['observed_outcomes'][condition]
            rate=f"{record['ratio']:.1%}" if record['ratio'] is not None else 'N/A'
            print(f"{args.task}/{condition}: success {record['successes']}/{record['completed']} ({rate}); censored={record['censored']}; selected development scenes",flush=True)
        print('PAIRED_GATE',result['gate_passed'],'INTERFACE',result['interface_passed'],'RUN',run,flush=True)
        if not result['interface_passed']:raise RuntimeError('Pairing/interface/numeric evidence failed; inspect result')
    except BaseException:
        if not (run/'failure.json').exists():write_json(run/'failure.json',dict(status='failed_or_interrupted',completed_records=len(episodes),traceback=traceback.format_exc()))
        raise
    finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(__doc__)
    from .evaluate_b2 import TASKS
    parser.add_argument('--task',choices=TASKS);parser.add_argument('--training-run',required=True,type=Path)
    parser.add_argument('--gpu',default='0');parser.add_argument('--tokenizer-path',type=Path)
    parser.add_argument('--prepare-protocol',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-chunks',type=int);parser.add_argument('--scene-timeout',default=1800,type=int)
    args=parser.parse_args();args.training_run=output_path(args.training_run)
    if not args.gpu.isdigit() or args.scene_timeout<60 or (args.max_chunks is not None and not 1<=args.max_chunks<=4):parser.error('Require GPU>=0, timeout>=60, optional1..4 chunk smoke')
    if args.prepare_protocol:
        if args.task or args.dry_run or args.max_chunks:parser.error('--prepare-protocol is a separate CPU phase')
        prepare_protocol(args.training_run);return
    if args.task is None:parser.error('--task required')
    from .evaluate_structured_low_storage import _terminate
    from .native_entry_storage import NativeEntryStorage
    storage=NativeEntryStorage()
    if not args.dry_run:
        atexit.register(storage.cleanup);signal.signal(signal.SIGTERM,_terminate)
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['JAX_PLATFORMS']='cpu'
    execute(args,storage)


if __name__=='__main__':main()
