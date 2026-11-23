"""Actual1GiB stream allocations; no model inference or artificial GPU work."""
from pathlib import Path
import copy,json,hashlib,weakref,gc
import torch
from native_expert_full_feedback_v1.stream_inputs import StreamInputs
from .reclaim import reclaim,rss_bytes
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);before=torch.random.get_rng_state().clone();keep=torch.arange(64,dtype=torch.float32);keep_before=keep.clone();pointer=keep.data_ptr()
 spec=read(T/'outputs/native_expert_feedback_v1/protocol.json');job=copy.deepcopy(spec['jobs'][0])
 for segment in job['segments']:segment['endpoint']=segment['boundary']['observation']
 inputs=StreamInputs(spec['input_sha256'],raw_limit_bytes=512*2**20,normalized_limit_bytes=4*2**30)
 baseline=rss_bytes();counts=inputs.prepare(job,1024);peak=rss_bytes();refs=[weakref.ref(v)for v in inputs.cache.values()]+[weakref.ref(inputs)]
 try:reclaim(refs)
 except ValueError:pass
 else:raise AssertionError('Live owner accepted as complete')
 inputs.clear();del inputs;gc.collect();after_clear=rss_bytes();result=reclaim(refs)
 assert keep.data_ptr()==pointer and torch.equal(keep,keep_before)and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 report=dict(passed=True,actual_saved_job=job['job_id'],actual_input_counts=counts,RSS_baseline_bytes=baseline,RSS_after_preparation_bytes=peak,RSS_after_clear_bytes=after_clear,reclaim=result,live_owner_refused=True,live_tensor_pointer_and_values_unchanged=True,CPU_RNG_unchanged=True,GPU_forwards=0,normal_GPU_exit_still_unverified=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 out=T/'outputs/native_completed_memory_v1';out.mkdir(exist_ok=True)
 with(out/'CPU_real_stream_reclaim_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('REAL_COMPLETED_STREAM_CPU_RECLAIM',report,flush=True)
if __name__=='__main__':main()
