"""CPU-only request schema/RNG/byte binding plus exact original scientific AST."""
import ast,random
import numpy as np,torch
from .common import C,T,O,read,write,sha
from .prefetch import PrefetchClient
from .semantic import content_key
def main():
 f=next((T/'outputs/native_cross_task100_v3/batches/002/runs/f06bbbf3b5f44bb498e919d679727527/semantic/requests').glob('*.json'));row=read(f)
 with np.load(row['frame_file'],allow_pickle=False)as z:observed=dict(rgb=tuple(z['cam'+str(i)].copy()for i in range(3)),command_state=z['command_state'].copy())
 root=O/'CPU_prefetch_fixture';client=PrefetchClient(root);text=row['instruction'];py=random.getstate();ns=np.random.get_state();ts=torch.get_rng_state().clone();key=client.submit(observed,text);assert key==row['content_key']
 assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1])and torch.equal(ts,torch.get_rng_state())
 req=read(root/'requests'/(key+'.json'));assert set(req)=={'frame_file','frame_sha256','instruction','content_key'}and sha(req['frame_file'])==req['frame_sha256']
 assert client.submit(observed,text)==key and len(list((root/'requests').glob('*.json')))==1
 answer=dict(content_key=key,CUDA_initialized=False,CPU_RNG_preserved=True,available=False,reason='software_fixture_only',models_state_sha256={'fixture':'not_a_model_result'},cold_CPU_inference_seconds=0.)
 write(root/'answers'/(key+'.json'),answer);expected,masks,report=client.at_initial(observed,text,query=0,tick=0);assert masks is None and not report['available']
 assert all(a.tobytes()==b.tobytes()for a,b in zip(expected['rgb'],observed['rgb'],strict=True))
 changed=dict(observed,command_state=observed['command_state'].copy());changed['command_state'][0]+=1
 for obs,txt in [(changed,text),(observed,text+' changed')]:
  try:client.at_initial(obs,txt,query=0,tick=0)
  except AssertionError:pass
  else:raise AssertionError('Mismatched prefetch accepted')
 def body(path,name):return next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(body(C/'runtime.py','run_case'))==ast.dump(body(T/'native_cross_task100_v9/runtime.py','run_case'))
 assert (C/'screen.py').read_text()==(T/'native_initial_prefetch_probe_v3/screen.py').read_text()
 for name in ['semantic.py','semantic_worker.py']:
  assert ast.dump(ast.parse((C/name).read_text().replace('native_cross_task100_v11','native_cross_task100_v9')))==ast.dump(ast.parse((T/'native_cross_task100_v9'/name).read_text()))
 assert not torch.cuda.is_initialized()
 write(O/'CPU_prefetch_checks.json',dict(passed=True,same_four_field_RGB_command_language_request_schema=True,duplicate_submission_once=True,wrong_command_and_language_fail_closed=True,original_at_initial_reused=True,original_entire_run_case_AST_exact=True,original_two_thread_CPU_inference_AST_exact=True,screen_identical_to_actual_eight_input_GPU_probe=True,submit_preserves_RNG=True,GPU_calls=0,source_sha256={str(p):sha(p)for p in C.glob('*.py')}))
 print('ASYNC_Q0_PREFETCH_CPU_CONTRACT_PASS',flush=True)
if __name__=='__main__':main()
