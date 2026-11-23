"""Dynamically refill task/split lanes; original expert collector and data audit."""
import argparse
from contextlib import ExitStack
import fcntl
import functools
import gc
import importlib
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import traceback
import uuid

PREPARED=None
PREPARED_PENDING=None
PREPARED_STREAM=None
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def jobs(protocol):
    return [(task,split) for split in ('adapter_fit','adapter_validation') for task in protocol['tasks']]


def route_planners(cls,coordinator):
    """Each robot retains its own original planners; only their execution is serialized."""
    for name in ('plan_path','plan_batch'):
        original=getattr(cls,name)
        def make(fn):
            @functools.wraps(fn)
            def invoke(self,*args,**kwargs):
                lane=coordinator.rng.current()
                if lane is None:raise RuntimeError('Planner call outside collection lane')
                def execute():return fn(self,*args,**kwargs)
                execute._cx_priority=0;execute._cx_stage="expert_"+fn.__name__
                return coordinator.call(lane,execute)
            return invoke
        setattr(cls,name,make(original))


def prepare_task(task,lane,c,native):
    original_setup=task.setup_demo
    def setup(*args,**kwargs):
        def create():
            with c.scene_gate.write():
                result=original_setup(*args,**kwargs)
                from collection_native_step import install as install_step
                install_step(task,c.scene_gate,c.collection_step)
                from collection_native_control import install as install_control
                install_control(task,c.scene_gate,c.collection_control)
                from native_render import install_task
                install_task(task,c.scene_gate,native,c.note)
                return result
        create._cx_priority=2 if kwargs.get("save_data") else 5
        create._cx_stage="replay_scene_setup"if kwargs.get("save_data")else"expert_scene_setup"
        return c.call(lane,create)
    task.setup_demo=setup
    original_close=task.close_env
    def close(*args,**kwargs):
        def destroy():
            with c.scene_gate.write():return original_close(*args,**kwargs)
        return c.call(lane,destroy)
    task.close_env=close
    # Original observation internals/native capture remain in the owning lane.
    # Scene teardown/setup must not overlap a renderer operation in another lane.
    for name in ('get_obs','_update_render'):
        original=getattr(task,name)
        def make(fn):
            @functools.wraps(fn)
            def invoke(*args,**kwargs):
                c.check_cancel()
                with c.scene_gate.read():return fn(*args,**kwargs)
            return invoke
        setattr(task,name,make(original))
    return task


def prepare_cpu(run,p,lanes,pending,stream=False):
    global PREPARED_STREAM
    from cx_tta.paths import write_json
    from cx_tta.audit import digest
    from cx_tta.native_entry_storage import NativeEntryStorage
    from cx_tta.recording import instrument_runtime
    pool=run/'collection_pools'/('pool_'+uuid.uuid4().hex[:12]);pool.mkdir(parents=True)
    sources={x.name:digest(x) for x in (HERE/'collection_pool.py',HERE/'adapter_stream_cache.py',HERE/'adapter_stream_host.py',HERE/'stream_disk_seed.py',HERE/'adapter_cpu_preload.py',HERE/'adapter_cache_cpu.py',HERE/'collection_capture_gate.py',HERE/'collection_gc_schedule.py',HERE/'collection_optional_open3d.py',HERE/'collection_imports_cpu.py',HERE/'collection_native_step.py',HERE/'collection_step.cpp',HERE/'collection_native_control.py',HERE/'collection_control.cpp',HERE/'collection_prepared.py',HERE/'collection_cpu_io.py',HERE/'concurrent_core.py',HERE/'concurrent_rng.py',HERE/'model_service.py',HERE/'native_render.py',HERE/'scene_gate.py')}
    write_json(pool/'manifest.json',dict(protocol_sha256=digest(run/'protocol.json'),deployment_sources=sources,pending=pending,lanes=lanes,scientific_sources_unchanged=True,original_plan_replay_capture_audit=True))
    storage=NativeEntryStorage(runtime_source=Path(p['runtime_source']))
    runtime=storage.prepare_runtime(pool);count=instrument_runtime(runtime)
    write_json(pool/'instrumentation.json',dict(counted_step_calls=count))
    storage.configure_runtime(pool,runtime,Path(p['tokenizer']))
    # All are imports/private copies only: never construct a SAPIEN scene or planner.
    from cx_tta import collect_smoke,stacking_recording,collect_stacking_experts_v2,collect_stacking_resumable,stacking_training_data,evaluate_b2
    from concurrent_core import Coordinator
    from concurrent_rng import Lane
    from model_service import ModelService
    from collection_cpu_io import CpuIO
    from collection_imports_cpu import preload
    imports=preload(p["runtime_source"])
    write_json(pool/"cpu_imports.json",imports)
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    if stream:
        from adapter_stream_cache import prepare as prepare_stream
        PREPARED_STREAM=prepare_stream(run,p,pool)
        from adapter_cpu_preload import prepare as prepare_policy_cpu
        prepare_policy_cpu(pool,PREPARED_STREAM)
    return pool,storage,runtime,sources


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--lanes',type=int,default=3);ap.add_argument('--plan',action='store_true');ap.add_argument('--stream-cache',action='store_true');a=ap.parse_args()
    if not 1<=a.lanes<=6:raise ValueError('Only bounded admission candidate 1..6 lanes')
    from task_adapters.protocol import load
    from task_adapters.collection import resume_state,audit
    from cx_tta.paths import write_json,replace_json
    from cx_tta.audit import digest
    run=a.run.resolve();p=load(run)
    if PREPARED_PENDING is not None:pending=PREPARED_PENDING
    else:
        for task in p['tasks']:audit(run,p,task,'smoke')
        pending=[]
        for task,split in jobs(p):
            folder=run/'collections'/task/split
            state,_=resume_state(folder,p,task,split)
            if state['remaining']:pending.append((task,split))
            else:audit(run,p,task,split)
    print('COLLECTION_POOL_PLAN',json.dumps(dict(pending=pending,lanes=a.lanes,protocol_sha256=digest(run/'protocol.json'),original_collect=True,independent_planners=True)),flush=True)
    if a.plan or not pending:return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('CX_TARGET_GPU_UUID'):raise RuntimeError('GPU UUID pin missing')
    from concurrent_core import Coordinator
    from concurrent_rng import Lane
    from model_service import ModelService
    from native_physics import load_native
    from collection_cpu_io import CpuIO
    from cx_tta.native_entry_storage import NativeEntryStorage
    from cx_tta.recording import instrument_runtime
    from cx_tta.collect_smoke import make_config
    from cx_tta.stacking_recording import install_recording
    from cx_tta.collect_stacking_experts_v2 import export_recording
    from cx_tta.collect_stacking_resumable import collect
    from cx_tta.stacking_training_data import qualify_capture
    from cx_tta.evaluate_b2 import seed_all
    import torch
    import torch.multiprocessing as mp
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Require one visible GPU')
    torch.set_num_threads(1);mp.set_start_method('spawn',force=True)
    proof=json.loads((HERE/'reports/target_collection_cpu_io_20261023.log').read_text().splitlines()[-1])
    if not proof['passed'] or proof['helper_sha256']!=digest(HERE/'collection_cpu_io.py'):raise RuntimeError('CPU IO equivalence proof stale')
    io=CpuIO(workers=1)
    c=Coordinator(cuda=True)
    c.service=ModelService(c);native=load_native()
    from collection_native_step import load as load_step
    from collection_native_control import load as load_control
    c.collection_step=load_step();c.collection_control=load_control();c.rng.install();retained=[]
    def stop(*_):c.cancel.set();raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if PREPARED is not None:
        pool,storage,runtime,sources=PREPARED
    else:pool,storage,runtime,sources=prepare_cpu(run,p,a.lanes,pending,a.stream_cache)
    from collection_optional_open3d import install as install_optional
    optional_proof=json.loads((HERE/"reports/target_collection_optional_open3d_cpu_20261024.json").read_text())
    if not optional_proof["passed"]or optional_proof["helper_sha256"]!=digest(HERE/"collection_optional_open3d.py"):raise RuntimeError("Optional dependency CPU proof stale")
    optional_module=install_optional(runtime)
    import envs.robot.robot as robot
    print("COLLECTION_OPEN3D_DEFERRED",optional_module._cx_real is None,flush=True)
    from envs.utils.create_actor import UnStableError
    # Original gripper interpolation is CPU-only and independent of planner state.
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(robot.CuroboPlanner.plan_grippers)))
    if any(isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=="self"for n in ast.walk(tree)):raise RuntimeError("Gripper interpolation contract changed")
    from collection_capture_gate import install as install_capture_gate
    restore_capture_gate=install_capture_gate(robot.CuroboPlanner,c)
    route_planners(robot.CuroboPlanner,c)
    stream_model=None
    if a.stream_cache:
        if PREPARED_STREAM is None:raise RuntimeError('Stream cache CPU preflight missing')
        from adapter_cpu_preload import initialize
        stream_model=initialize(PREPARED_STREAM,c,pool)
    work=queue.Queue()
    for job in pending:work.put(job)
    failures=[];completed=[];state_lock=threading.Lock()
    first_capture=[threading.Event()for _ in range(a.lanes)]
    def save():replace_json(pool/'status.json',dict(completed=completed,pending=work.qsize(),failures=failures,pid=os.getpid()))
    def worker(index):
        lane=Lane('collect/'+str(index));task=None
        try:
            if index:
                while not first_capture[0].wait(.05):c.check_cancel()
            with c.rng.bind(lane):
                while not c.cancel.is_set():
                    try:task_name,split=work.get_nowait()
                    except queue.Empty:return
                    c.note('collection_job_begin',lane=lane.name,task=task_name,split=split)
                    folder=run/'collections'/task_name/split;folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'writer.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        manifest=dict(task=task_name,split=split,protocol_sha256=digest(run/'protocol.json'),episodes=p['splits'][split]['episodes'],video=False)
                        if (folder/'manifest.json').exists():
                            if json.loads((folder/'manifest.json').read_text())!=manifest:raise ValueError('Collection manifest changed')
                        else:write_json(folder/'manifest.json',manifest)
                        state,_=resume_state(folder,p,task_name,split)
                        session=folder/'sessions'/('session_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
                        write_json(session/'manifest.json',dict(manifest,initial=state,collection_pool=str(pool),deployment_sources=sources))
                        args=argparse.Namespace(task=task_name,task_config=p['task_config'],split=split,protocol=run/'protocol.json',gpu='0',min_free_gb=10)
                        config=make_config(runtime,args,session);config.update(eval_mode=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
                        if config['save_freq']!=15 or config['data_type']['rgb'] is not True:raise ValueError('Original recording contract changed')
                        write_json(session/'resolved_config.json',config)
                        def construct():
                            obj=getattr(importlib.import_module('envs.'+task_name),task_name)();install_recording(obj,effect_schema=True)
                            obj=prepare_task(obj,lane,c,native)
                            picture=obj._take_picture
                            def capture():
                                result=picture()
                                if obj.save_data and obj.FRAME_IDX>0:first_capture[index].set()
                                return result
                            obj._take_picture=capture
                            return obj
                        construct._cx_priority=6;construct._cx_stage="task_class_import"
                        task=c.call(lane,construct)
                        def seed(value):return c.call(lane,seed_all,value)
                        result=collect(task,config,args,p,folder,session,state,seed,UnStableError,io.export,io.qualify)
                        replace_json(folder/'result.json',result)
                        if result['status']!='collection_complete':raise RuntimeError('Original expert seed window exhausted')
                        audit(run,p,task_name,split)
                        write_json(session/'source_runtime_audit.json',storage.verify_source())
                        load(run)
                        # Clear wrapper closures/scene/planner cycles on their CUDA owner.
                        current=task
                        def release():
                            with c.scene_gate.write():
                                if work.empty():retained.append(current)
                                else:current.__dict__.clear();gc.collect();torch.cuda.empty_cache()
                        c.call(lane,release);task=None;current=None
                        with state_lock:completed.append([task_name,split]);save()
                        c.note('collection_job_complete',lane=lane.name,task=task_name,split=split)
        except BaseException:
            c.cancel.set()
            with state_lock:failures.append(traceback.format_exc());save()
            print('COLLECTION_POOL_FAILURE',failures[-1],flush=True)
    threads=[threading.Thread(target=worker,args=(i,),daemon=True,name='collection-'+str(i)) for i in range(min(a.lanes,len(pending)))]
    collection_done=threading.Event();cache_thread=None
    if a.stream_cache:
        def cache_worker():
            try:
                from adapter_stream_cache import work as cache_work
                cache_work(run,p,PREPARED_STREAM,c,pool,collection_done,stream_model)
            except BaseException:
                c.cancel.set()
                with state_lock:failures.append(traceback.format_exc());save()
                print('COLLECTION_STREAM_FAILURE',failures[-1],flush=True)
        cache_thread=threading.Thread(target=cache_worker,daemon=True,name='original-training-cache')
    save()
    if cache_thread:cache_thread.start()
    for thread in threads:thread.start()
    while any(t.is_alive() for t in threads)or(cache_thread and cache_thread.is_alive()):
        if not any(t.is_alive()for t in threads):collection_done.set()
        c.service.process_one(.01)
        for thread in threads:thread.join(.001)
    if failures:raise RuntimeError(failures[0])
    # Every completed split was audited already; host repeats the global audit
    # after GPU release, avoiding an idle GPU context during CPU-only readback.
    io.close()
    write_json(pool/'source_runtime_audit.json',storage.verify_source());load(run)
    c.rng.uninstall()
    print('COLLECTION_POOL_ALL_COMPLETE',pool,flush=True)
    sys.stdout.flush();sys.stderr.flush()
    # Original Torch compiler worker teardown can retain a GPU context; all data,
    # full audits and source checks are already durable. Host verifies GPU release.
    os._exit(0)

if __name__=='__main__':main()
