"""Spawn-only CPU export/readback using the original scientific implementations."""
import os
from pathlib import Path
from types import SimpleNamespace

def initialize_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['NVIDIA_VISIBLE_DEVICES']='void'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'

def export_snapshot(snapshot,runtime,task_name,info,seed,index):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU export must hide CUDA')
    from cx_tta.collect_stacking_experts_v2 import export_recording
    registry=snapshot.pop('registry')
    task=SimpleNamespace(**snapshot,_cx_actor_registry=SimpleNamespace(metadata=lambda:registry))
    return str(export_recording(task,runtime,task_name,info,seed,index))

def qualify(path,task,seed,split,sha):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU readback must hide CUDA')
    from cx_tta.stacking_training_data import qualify_capture
    return qualify_capture(Path(path),task,seed,split,sha)

class CpuIO:
    def __init__(self,workers=4):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_cpu)
    def export(self,task,runtime,task_name,info,seed,index):
        # Serialize the exact original dense trace and immutable per-scene metadata.
        # No SAPIEN/robot/actor/CUDA object crosses the process boundary.
        snapshot=dict(save_dir=task.save_dir,folder_path=task.folder_path,_cx_dense=task._cx_dense,
                      cx_effect_enabled=task.cx_effect_enabled,registry=task._cx_actor_registry.metadata())
        return Path(self.pool.submit(export_snapshot,snapshot,runtime,task_name,info,seed,index).result())
    def qualify(self,*args):return self.pool.submit(qualify,*args).result()
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
