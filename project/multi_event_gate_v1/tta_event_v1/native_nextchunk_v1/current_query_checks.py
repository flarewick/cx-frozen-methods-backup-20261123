"""Current-RGB API on real cached inputs; real frozen adapter/FK, SPY base only.
No actual VLA weights forward, CUDA or simulator continuation claim.
"""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from cx_tta.eval_diagnostics import array_hash
from native_stage_v1.cx_tta_stage.query import build_head
from query_current import query_current
torch.set_num_threads(2)
C=Path(__file__).resolve().parent;T=C.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return{k:z[k].copy() for k in z.files}
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid current image/query accepted')
class SpyBase:
    def __init__(self,base):self.base=base;self.inputs=[]
    def infer(self,obs,*,noise=None):
        assert noise is None
        self.inputs.append({k:v.copy() for k,v in obs['images'].items()})
        return dict(actions=self.base.copy())
def run():
    p=read(T/'outputs/native_stage_v1/protocol.json')
    head,bots=build_head(p['training']);projection=load(p['projection'])
    jobs=read(T/'outputs/silhouette_feedback_cohort_v1/protocol.json')['jobs']
    records=[];checks=0
    for job in jobs[::8]:
        folder=Path(job['packet']).parent;initial=load(folder/'initial.npz')
        rgb=load(folder/'Z_reference_rgb.npz');history=load(folder/'observed_history.npz')
        current=dict(initial,rgb=tuple(rgb['initial_'+str(i)] for i in range(3)),observation_tick=0)
        current['images']={str(i):array_hash(im) for i,im in enumerate(current['rgb'])}
        spy=SpyBase(load(folder/'proposal.npz')['base'])
        result=query_current(spy,head,bots,projection,current,'Stack the objects.',None,decision_tick=0)
        order=('cam_high','cam_right_wrist','cam_left_wrist')
        for i,key in enumerate(order):
            assert spy.inputs[-1][key].tobytes()==current['rgb'][i].transpose(2,0,1).tobytes();checks+=1
        assert result['actor_images']==current['images'];checks+=1
        anchor_hash={str(i):array_hash(history['cam'+str(i)][0]) for i in range(3)}
        assert anchor_hash!=current['images'];checks+=1
        before=len(spy.inputs)
        reject(lambda:query_current(spy,head,bots,projection,dict(current,images=anchor_hash),'Stack the objects.',None,decision_tick=0))
        reject(lambda:query_current(spy,head,bots,projection,current,'Stack the objects.',None,decision_tick=1))
        assert len(spy.inputs)==before;checks+=3
        poisoned=dict(current,future_rgb=np.full((3,7,7,3),np.nan),task_id='FORBIDDEN_DIAGNOSTIC_METADATA',seed=-99)
        repeat=query_current(spy,head,bots,projection,poisoned,'Stack the objects.',None,decision_tick=0)
        assert repeat['base'].tobytes()==result['base'].tobytes()
        assert repeat['raw_delta'].tobytes()==result['raw_delta'].tobytes();checks+=2
        records.append(dict(case_id=job['case_id'],current_images_differ_from_initial_actor_anchor=True,
                   actor_exact_current_RGB=True,head_delta_repeated_exact=True))
    report=dict(passed=True,checks=checks,records=records,real_frozen_adapter_and_FK_used=True,
         base_policy_is_spy_not_VLA=True,no_CUDA_or_simulator_reference_claim=True,NOT_GPU_READY=True,
         source_sha256={str(C/n):sha(C/n) for n in ('query_current.py','current_query_checks.py')})
    out=T/'outputs/native_nextchunk_v1/current_query_cpu_checks.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('CURRENT_RGB_QUERY_CPU_CHECKS_PASS',checks,'SPY_BASE_ONLY',flush=True)
if __name__=='__main__':run()
