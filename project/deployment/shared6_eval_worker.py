"""Run original trained-adapter closures, full100 per condition, on one GPU."""
from contextlib import ExitStack
import argparse,copy,fcntl,json,os,sys,threading,time,traceback,uuid
from pathlib import Path
D=Path(__file__).resolve().parent
PREPARED=None
KEEP=[]

def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--task',required=True);p.add_argument('--training-run',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--bundle',type=Path);p.add_argument('--lanes',type=int,default=4);p.add_argument('--episodes',type=int,default=100);p.add_argument('--mode',choices=('plan','run'),default='plan')
    return p.parse_args(argv)

def prepare_cpu(a):
    from shared6_eval_backend import make_plan,check_bundle,sources
    from cx_tta.native_entry_storage import NativeEntryStorage
    from cx_tta.paths import write_json
    from adapter_cache_cpu import load_cpu_policy,state_digest
    import torch
    plan=make_plan(a.task,a.training_run,a.gpu_uuid,a.lanes,a.episodes)
    training_plan=json.loads((a.training_run/'training_plan.json').read_text())
    from shared6_screen_reference import load_references
    screen_references=load_references(plan)
    if a.bundle:check_bundle(a.bundle,training_plan,a.gpu_uuid)
    for job in plan['jobs']:
        for name,value in job['manifest']['source_sha256'].items():
            from shared6_eval_backend import sha
            if sha(name)!=value:raise RuntimeError('Original preflight source changed')
    lanes=[];master=None
    for job in plan['jobs']:
        if job['complete']:continue
        for index in range(a.lanes//2):
            session=Path(job['run'])/'sessions'/('process_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
            for folder in('queries','onsets'):(session/folder).mkdir()
            storage=NativeEntryStorage(runtime_source=Path(job['manifest']['runtime_source']));runtime=storage.prepare_runtime(session)
            write_json(session/'manifest.json',dict(job['manifest'],initial_completed=job['completed'],cpu_prepared=True))
            if master is None:storage.configure_runtime(session,runtime,Path(job['tokenizer']));master=storage
            lanes.append(dict(job=job,index=index,session=session,storage=storage,runtime=runtime))
    if not lanes:return dict(plan=plan,lanes=[],master=None,policy=None,training_plan=training_plan)
    proof=json.loads([l.split(' ',1)[1]for l in(D/'reports/target_adapter_cache_cpu_arrow_nvml_20261024.log').read_text().splitlines()if l.startswith('ADAPTER_CACHE_CPU_PROOF ')][-1])
    from shared6_eval_backend import sha
    if not proof['passed']or proof['helper_sha256']!=sha(D/'adapter_cache_cpu.py'):raise RuntimeError('CPU loading proof stale')
    policy,info=load_cpu_policy(verify_state=True);digest=state_digest(policy._model)
    if digest!=proof['converted_state_sha256']or torch.cuda.is_initialized():raise RuntimeError('CPU frozen model differs or touched CUDA')
    write_json(lanes[0]['session']/'cpu_prepared_model.json',dict(passed=True,cpu_only=True,state_sha256=digest,source_sha256=sources(),**info))
    return dict(plan=plan,lanes=lanes,master=master,policy=policy,training_plan=training_plan,screen_references=screen_references)

class Captured(Exception):
    def __init__(self,callbacks):self.callbacks=callbacks

def capture(execute,*args):
    from cx_tta import evaluation_resume
    previous=evaluation_resume.stream
    def stream(run,manifest,screen,rollout,update,announce):
        values={}
        for callback in(screen,rollout,update,announce):
            for name,cell in zip(callback.__code__.co_freevars,callback.__closure__):values[name]=cell.cell_contents
        values.update(screen=screen,rollout=rollout,update=update,announce=announce);raise Captured(values)
    evaluation_resume.stream=stream
    try:execute(*args)
    except Captured as result:return result.callbacks
    finally:evaluation_resume.stream=previous
    raise RuntimeError('Original evaluation stream boundary changed')

def run_gpu(a,prepared):
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    from task_adapters import evaluate as original_evaluate,eval_resume as original
    from shared6_eval_backend import restore_cache,live_fingerprint,sha
    from shared6_eval_core import Shared6Coordinator,configure
    from adapter_cache_cpu import move_to_gpu
    from model_service import ModelService
    from concurrent_rng import Lane
    from concurrent_queue import EpisodeQueue
    from native_generic_physics import load_native
    from cx_tta.paths import write_json,replace_json
    from cx_tta import evaluate_b2,recovery_contract
    from cx_tta.native_entry_storage import NativeEntryStorage
    plan=prepared['plan'];master=prepared['master'];lanes=prepared['lanes']
    if not lanes:return
    c=Shared6Coordinator(cuda=True);c.service=ModelService(c);c.native=load_native();KEEP.extend([c,prepared]);c.rng.install()
    restored=restore_cache(a.bundle,prepared['training_plan'],a.gpu_uuid)
    from autotune_relocation import AutotuneRelocation
    relocation=AutotuneRelocation(os.environ['TORCHINDUCTOR_CACHE_DIR']);relocation.install();KEEP.append(relocation)
    c.base=move_to_gpu(prepared['policy']);c.base_factory=lambda:c.base
    from shared_weights import versions
    c.weight_versions=versions(c.base._model);c.parity_path=lanes[0]['session']/'deployment_query_parity.json';c.screen_references=prepared['screen_references'];c.screen_parity_path=lanes[0]['session']/'deployment_screen_parity.json'
    write_json(lanes[0]['session']/'deployment_model_cache.json',dict(imported=restored,cpu_preloaded=True,gpu_uuid=a.gpu_uuid,sources=plan['source_sha256']))
    from collection_optional_open3d import install as optional
    optional(lanes[0]['runtime'])
    from observation_materialization import wrap as materialize
    evaluate_b2.step_chunk_official=materialize(evaluate_b2.step_chunk_official,c.note)
    previous_seed=evaluate_b2.seed_all
    def routed(function):
        def call(*args,**kwargs):
            lane=c.rng.current()
            if lane is None:raise RuntimeError('Original Torch RNG operation outside a scene lane')
            return c.call(lane,function,*args,**kwargs)
        return call
    evaluate_b2.seed_all=routed(previous_seed)
    rng_original={name:getattr(recovery_contract,name)for name in('rng_capture','rng_restore')}
    for name,function in rng_original.items():setattr(recovery_contract,name,routed(function))
    exported=[False]
    def export(result,seed,instruction):
        if exported[0]:return
        if not result.get('passed'):raise RuntimeError('Original L0/zero steady probe failed')
        from compiler_bundle import write_bundle
        before=torch.get_rng_state().clone();cuda=[s.clone()for s in torch.cuda.get_rng_state_all()];value=torch.compiler.save_cache_artifacts()
        if value is None:return
        data,info=value
        if not info.inductor_artifacts or not info.autotune_artifacts:return
        if not torch.equal(before,torch.get_rng_state())or any(not torch.equal(x,y)for x,y in zip(cuda,torch.cuda.get_rng_state_all())):raise RuntimeError('Compiler export changed RNG')
        probe=dict(seed=int(seed),instruction=instruction,common_rng_sha256=result['common_rng_sha256'],initial_native_sha256=result['initial_native_sha256'],base_values_sha256=sha_bytes(result['base_values'].tobytes()),passed=True)
        fingerprint=live_fingerprint(plan['jobs'][0]['manifest'],prepared['training_plan']);meta=write_bundle(master.run/'deployment_compiler_bundle',fingerprint,data,info,probe);exported[0]=True;c.note('original_shared6_compiler_bundle_exported',artifact_sha256=meta['artifact_sha256'])
    c.cache_export=export
    class Storage(NativeEntryStorage):
        def __init__(self,row):self.__dict__.update(row['storage'].__dict__);self._row=row
        def prepare_runtime(self,run):
            if Path(run)!=self._row['session']:raise RuntimeError('CPU/GPU private session differs')
            return self._row['runtime']
        def configure_runtime(self,run,runtime,tokenizer):
            self.run=Path(run);self.scratch=master.scratch
            if self.run!=master.run:
                record=json.loads((master.run/'storage.json').read_text());record.update(shared_process_scratch_owner=str(master.run));write_json(self.run/'storage.json',record);write_json(self.run/'compiler_temp.json',dict(path=str(master.scratch/'tmp'),run=str(self.run)))
                write_json(self.run/'native_storage.json',dict(source=str(self.runtime_source),same_process_scratch_owner=str(master.run)))
        def cleanup(self):pass
    queues={};runtime_lanes=[];errors=[]
    with ExitStack()as locks:
        for job in plan['jobs']:
            lock=locks.enter_context((original.WORKSPACE_ROOT/'outputs/evaluation_locks'/(job['identity']+'.lock')).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in lanes:
            job=row['job'];lane=Lane(job['condition']+'/'+str(row['index']));args=original.options(['--task',a.task,'--training-run',str(a.training_run),'--condition',job['condition'],'--episodes','100','--seed','0','--task-config','demo_randomized']);args.dataset_run=Path(job['run']);args.initial_completed=job['completed'];storage=Storage(row)
            with c.rng.bind(lane),c.rng.torch_scope(lane):callbacks=capture(original_evaluate.execute,args,job['protocol'],Path(job['reference']),Path(job['tokenizer']),job['manifest'],storage,row['session'])
            if not runtime_lanes:
                import envs.robot.robot as robot
                from collection_capture_gate import install
                KEEP.append(install(robot.CuroboPlanner,c))
                from shared6_planner_pool import install as pool
                KEEP.append(pool(c))
            callbacks=configure(callbacks,c,lane);runtime_lanes.append((row,lane,callbacks,storage))
            if job['condition']not in queues:queues[job['condition']]=EpisodeQueue(Path(job['run']),job['manifest'],callbacks['update'],callbacks['announce'])
        def work(row,lane,callbacks,storage):
            try:
                with c.rng.bind(lane):
                    queues[row['job']['condition']].work(callbacks,c.cancel)
                    for name,expected in row['job']['manifest']['source_sha256'].items():
                        if sha(name)!=expected:raise RuntimeError('Original scientific source changed during evaluation')
                    for name,expected in plan['source_sha256'].items():
                        if sha(D/name)!=expected:raise RuntimeError('Deployment source changed during evaluation')
                    write_json(row['session']/'source_runtime_audit.json',storage.verify_source())
            except BaseException:
                c.cancel.set();message=traceback.format_exc();errors.append(message);replace_json(row['session']/'failure.json',dict(complete=False,traceback=message));print('SHARED6_LANE_FAILURE',lane.name,message,flush=True)
        threads=[threading.Thread(target=work,args=row,name=row[1].name,daemon=True)for row in runtime_lanes]
        for thread in threads:thread.start()
        while any(t.is_alive()for t in threads):
            c.service.process_one(.01)
            for thread in threads:thread.join(.001)
        if errors:raise RuntimeError(errors[0])
        c.verify_weights()
        for job in plan['jobs']:
            result=original.final_result(Path(job['run']),job['manifest']);replace_json(Path(job['run'])/'result.json',result);replace_json(Path(job['run'])/'progress.json',dict(phase='finished',**result));print('SHARED6_CONDITION_COMPLETE',job['condition'],json.dumps(result),flush=True)
        print('SHARED6_ORIGINAL_TWO_CONDITION_ALL_COMPLETE',a.task,flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)

def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()

def main():
    a=arguments()
    if a.mode=='plan':
        from shared6_eval_backend import make_plan
        print('SHARED6_PLAN',json.dumps(make_plan(a.task,a.training_run,a.gpu_uuid,a.lanes,a.episodes)),flush=True)
    else:
        if PREPARED is None:raise RuntimeError('GPU worker requires a proven CPU prelude')
        run_gpu(a,PREPARED)
if __name__=='__main__':main()
