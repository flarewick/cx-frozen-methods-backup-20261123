"""CPU control-flow and true CPU RNG draw sequencing; no CUDA/ref-policy claim."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json
import torch
from query_ledger import QueryLedger
torch.set_num_threads(2)
def sha(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid ledger transition accepted')
def draw(g,l,q,t):
    l.permit_query(q,t)
    before=sha(g.get_state());a=torch.randn((32,14),generator=g)
    after=sha(g.get_state());l.record_draw(q,t,sha(a),before,after,actual_sampler_call=True)
    return a
def run():
    checks=0
    for branch in ('Z_reference','C_original','Z_sham'):
        ledger=QueryLedger(branch);generator=torch.Generator().manual_seed(20261027)
        reference=torch.Generator().manual_seed(20261027)
        for q in range(3):
            a=draw(generator,ledger,q,q*100)
            assert torch.equal(a,torch.randn((32,14),generator=reference));checks+=1
            rejected(lambda:draw(generator,ledger,q+1,q*100));checks+=1
            assert torch.equal(generator.get_state(),reference.get_state());checks+=1
            ledger.finish_chunk(32,end_tick=(q+1)*100,requested95_arrived=q==2,origin=q==2)
        assert ledger.require_following_query(300)==3;checks+=1
        actual=draw(generator,ledger,3,300)
        assert torch.equal(actual,torch.randn((32,14),generator=reference));checks+=1
    bad=QueryLedger('bad')
    rejected(lambda:bad.record_draw(0,0,'a'*64,'b'*64,'b'*64,actual_sampler_call=True));checks+=1
    rejected(lambda:bad.record_draw(0,0,'a'*64,'b'*64,'c'*64,actual_sampler_call=False));checks+=1
    rejected(lambda:bad.require_following_query(0));checks+=1
    bad.record_draw(0,0,'a'*64,'b'*64,'c'*64,actual_sampler_call=True)
    bad.finish_chunk(22,end_tick=99,requested95_arrived=True,origin=True)
    rejected(lambda:bad.require_following_query(100));checks+=1
    rejected(lambda:bad.record_draw(1,100,'a'*64,'b'*64,'c'*64,actual_sampler_call=True));checks+=1
    unknown=QueryLedger('unknown95')
    unknown.record_draw(0,0,'a'*64,'b'*64,'c'*64,actual_sampler_call=True)
    unknown.finish_chunk(32,end_tick=100,requested95_arrived=False,origin=True)
    rejected(lambda:unknown.permit_query(1,100));checks+=1
    clock=QueryLedger('clock')
    clock.record_draw(0,0,'a'*64,'b'*64,'c'*64,actual_sampler_call=True)
    clock.finish_chunk(32,end_tick=100)
    rejected(lambda:clock.permit_query(1,50));checks+=1
    # Cached replay silently restarts first sample; proper draw ledger cannot label
    # this equivalent to consuming all prior query draws.
    g=torch.Generator().manual_seed(7);first=torch.randn((32,14),generator=g)
    next_real=torch.randn((32,14),generator=g)
    restarted=torch.randn((32,14),generator=torch.Generator().manual_seed(7))
    assert torch.equal(first,restarted) and not torch.equal(next_real,restarted);checks+=1
    out=Path(__file__).resolve().parents[1]/'outputs/native_nextchunk_v1'
    out.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,checks=checks,scope='CPU ledger/control-flow and explicit CPU torch.Generator only. Runtime NOT integrated, actual base internal CPU/CUDA RNG/router/physics/reference parity unverified. NOT GPU READY or TTA evidence.',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (out/'cpu_ledger_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NEXTCHUNK_CPU_LEDGER_PASS',checks,'NOT_GPU_READY',flush=True)
if __name__=='__main__':run()
