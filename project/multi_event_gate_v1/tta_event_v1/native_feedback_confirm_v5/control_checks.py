"""CPU real callback/control contracts; no CUDA, real SAM2 forward or physics."""
from pathlib import Path
from types import SimpleNamespace
import ast,importlib,json
import numpy as np,torch
from cx_tta_feedback_confirm5.paths import OUTPUT,write,digest
from cx_tta_feedback_confirm5.capture import CurrentReadouts
from cx_tta_feedback_confirm5.sam2_core import OffloadedStream
from native_delayed_query_v1.boundary import native_continue
from query_object_v1.stream import video
from cx_tta_feedback_confirm5.model_math import sam2_profile
C=Path(__file__).resolve().parent

def run():
    torch.set_num_threads(2);checks=0
    for name in('current_case','parallel_runtime','preload','parent_callbacks','contract','actor_trace','live_audit','live_feedback'):
        importlib.import_module('cx_tta_feedback_confirm5.'+name);checks+=1
    recorder=SimpleNamespace(rows=[],dt=[],failed=False);reads=[];callbacks=[]
    def reader(tick,elapsed):reads.append((tick,elapsed));return dict(actual_tick=tick)
    def arrived(tick,elapsed,observation,rec):
        assert len(rec.rows)==tick and observation['actual_tick']==tick
        callbacks.append(tick)
    hook=CurrentReadouts(recorder,reader,on_arrival=arrived)
    for i in range(33):
        before=len(recorder.rows);recorder.rows.append({});recorder.dt.append(float(np.float32(.004)));hook.after(before)
    assert callbacks==[32];checks+=1
    a=hook.read(33);b=hook.read(33)
    assert a is b and callbacks==[32,33]and len(reads)==2;checks+=1
    assert native_continue(SimpleNamespace(take_action_cnt=31,step_lim=32,eval_success=False));checks+=1
    assert not native_continue(SimpleNamespace(take_action_cnt=32,step_lim=32,eval_success=False));checks+=1
    assert not native_continue(SimpleNamespace(take_action_cnt=31,step_lim=32,eval_success=True));checks+=1
    assert OffloadedStream.append is video.Stream.append;checks+=1
    for cls in(OffloadedStream,video.Stream):
        source=Path(video.__file__ if cls is video.Stream else C/'cx_tta_feedback_confirm5/sam2_core.py').read_text();tree=ast.parse(source)
        target=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls.__name__)
        init=next(n for n in target.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
        calls=[n for n in ast.walk(init)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='init_state']
        assert len(calls)==1
        flags={k.arg:ast.literal_eval(k.value)for k in calls[0].keywords}
        assert flags['offload_video_to_cpu']is True
        assert flags['offload_state_to_cpu']is(cls is OffloadedStream);checks+=1
    tree=ast.parse((C/'cx_tta_feedback_confirm5/current_case.py').read_text())
    assert not any(isinstance(n,ast.Attribute)and n.attr=='check_success'for n in ast.walk(tree));checks+=1
    guards=[n for n in ast.walk(tree)if isinstance(n,ast.If)and 'following_complete' in ast.unparse(n.test)]
    assert len(guards)==1 and ast.unparse(guards[0].test)=='following_complete and used < 5000 and native_continue(task)';checks+=1
    assert not torch.cuda.is_initialized();checks+=1
    previous=torch.get_float32_matmul_precision();rng=torch.random.get_rng_state().clone()
    torch.set_float32_matmul_precision('high')
    try:
        with sam2_profile():assert torch.get_float32_matmul_precision()=='highest';checks+=1
        assert torch.get_float32_matmul_precision()=='high';checks+=1
        try:
            with sam2_profile():raise ValueError('Fixture failure')
        except ValueError:pass
        assert torch.get_float32_matmul_precision()=='high'and torch.equal(rng,torch.random.get_rng_state());checks+=1
    finally:torch.set_float32_matmul_precision(previous)
    names=('control_checks.py','cx_tta_feedback_confirm5/capture.py','cx_tta_feedback_confirm5/current_case.py',
           'cx_tta_feedback_confirm5/sam2_core.py','cx_tta_feedback_confirm5/preload.py',
           'cx_tta_feedback_confirm5/parallel_runtime.py','cx_tta_feedback_confirm5/parent_callbacks.py',
           'cx_tta_feedback_confirm5/contract.py','cx_tta_feedback_confirm5/actor_trace.py','cx_tta_feedback_confirm5/model_math.py')
    write(OUTPUT/'control_checks.json',dict(passed=True,checks=checks,CPU_fixture_only=True,
        existing_readout_callback_once_per_arrived_frame=True,native_terminal_guard_and_fixed32_horizon=True,
        official_state_offload_storage_flag_only=True,actual_SAM2_numeric_equivalence_NOT_proven=True,
        source_sha256={str(C/n):digest(C/n)for n in names},GPU_ready=False,online_updates=0,TTA_utility_proven=False))
    print('LIVE_NATIVE_CONTROL_IMPORT_CPU_PASS',checks,flush=True)
if __name__=='__main__':run()
