"""Isolated two-candidate existing-scene diagnostic, original sampler/physics/journal."""
import ast,copy,functools,hashlib,inspect,json,textwrap
from pathlib import Path
D=Path(__file__).resolve().parent
FILES=('candidate_eval_runtime.py','candidate_eval_policy.py','candidate_eval_reference.py','candidate_eval_checkpoint.py')

def install():
    from shared6_eval_resume import install as resume
    resume()
    import shared6_eval_backend as backend
    if getattr(backend,'_candidate_installed',False):return
    from candidate_eval_checkpoint import CANDIDATES,candidate,STUDY
    from candidate_eval_reference import FILE,load,sha
    reference=load()
    for name in CANDIDATES:candidate(name)
    original_sources=backend.sources;original_decorate=backend.decorate
    def sources():return dict(original_sources(),**{name:sha(D/name) for name in FILES})
    backend.sources=sources
    def decorate(raw,uuid,lanes):
        if uuid!=reference['gpu_uuid']:raise RuntimeError('Exact original baseline comparison is pinned to original GPU UUID')
        result=original_decorate(raw,uuid,lanes)
        result['deployment_candidate_diagnostic']=dict(protocol='old_scene_two_optimized_candidates_v1',source_sha256={name:sources()[name] for name in FILES},old_scene_reference_sha256=sha(FILE),comparison='Development/regression on previously observed EVAL scenes, not independent final test',outcomes_imported=False)
        from cx_tta.evaluation_resume import identity
        result['evaluation_identity']=identity(result);return result
    backend.decorate=decorate
    from task_adapters import eval_resume as original
    old_options=original.options;old_preflight=original.preflight
    def options(argv=None):
        import sys
        values=list(sys.argv[1:] if argv is None else argv);external=None
        if '--condition' in values:
            pos=values.index('--condition')+1
            if values[pos] in CANDIDATES:external=values[pos];values[pos]='M_event_full'
        args=old_options(values)
        if external:args.condition=external
        return args
    original.options=options
    def preflight(args):
        protocol,ref,tokenizer,manifest=old_preflight(args)
        if args.task!='open_laptop' or args.condition not in CANDIDATES:raise ValueError('Only authorized two open_laptop candidates')
        payload,path,spec=candidate(args.condition)
        manifest.update(checkpoint_sha256=sha(path),checkpoint_caption=args.condition,training_tasks=payload['metadata']['job']['tasks'],
            candidate_stage='validation_selected_expert_initialization_scaled',comparison='Previously evaluated100scenes regression; no independent-test claim',
            candidate=dict(checkpoint=str(path),checkpoint_sha256=sha(path),selection=payload['metadata']['selection'],job=payload['metadata']['job'],alpha=spec['alpha'],gate_trained=False,TTA=False))
        for file in [FILE,STUDY/'protocol.json',STUDY/'audit.json',path,path.parent/'result.json',*[D/name for name in FILES]]:
            manifest['source_sha256'][str(file)]=sha(file)
        protocol['source_sha256']=manifest['source_sha256']
        from cx_tta.evaluation_resume import identity
        manifest['evaluation_identity']=identity(manifest)
        return protocol,ref,tokenizer,manifest
    original.preflight=preflight
    # Only replace the two condition literals; retain original plan construction.
    node=ast.parse(textwrap.dedent(inspect.getsource(backend.make_plan))).body[0]
    loops=[n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='condition']
    if len(loops)!=1 or ast.literal_eval(loops[0].iter)!=('Z_zero','M_event_full'):raise RuntimeError('Original plan condition boundary changed')
    loops[0].iter=ast.Tuple(elts=[ast.Constant(name) for name in CANDIDATES],ctx=ast.Load())
    ns=dict(backend.make_plan.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(D/'candidate_eval_runtime.py'),'exec'),ns);backend.make_plan=ns['make_plan']
    # Keep the live policy import in the admitted fresh GPU child, exactly
    # where original run_gpu imported it.
    import shared6_eval_worker as worker
    original_run_gpu=worker.run_gpu
    def run_gpu(*args,**kwargs):
        import task_adapters.policy as policy
        from candidate_eval_policy import CandidatePolicy
        policy.StackingPolicy=CandidatePolicy
        return original_run_gpu(*args,**kwargs)
    worker.run_gpu=run_gpu
    import shared6_screen_reference as screen
    def load_references(plan):
        return dict(expected=copy.deepcopy(reference['screens']),sources=[dict(manifest=str(FILE),manifest_sha256=sha(FILE),no_policy_outcomes=True)])
    screen.load_references=load_references
    backend._candidate_installed=True

if __name__=='__main__':
    # Install candidate imports only AFTER the original prelude hides CUDA and
    # sets Torch to one thread. Keep every original fork/admission check intact.
    from shared6_eval_prepared import main as original_prelude
    node=ast.parse(textwrap.dedent(inspect.getsource(original_prelude))).body[0]
    index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.set_num_threads')
    node.body[index+1:index+1]=ast.parse('_candidate_install()').body
    ns=dict(original_prelude.__globals__,_candidate_install=install)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(D/'candidate_eval_runtime.py'),'exec'),ns)
    ns['main']()
