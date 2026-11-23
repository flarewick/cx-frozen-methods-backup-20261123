"""Actual pinned sampler/infer bodies with fixture forward, real RngRouter/CPU RNG.
NOT a real VLA forward, CUDA equivalence or simulator continuation proof.
"""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
import jax
from typing_extensions import override
from concurrent_rng import RngRouter,Lane
from sampler_fork import FreshSamplerFork,torch_hash,environment_hash,full_hash
torch.set_num_threads(2)
C=Path(__file__).resolve().parent;T=C.parent
RT=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/policy/pi05/src/openpi')
SOURCES=(RT/'models_pytorch/pi0_pytorch.py',RT/'policies/policy.py')
MODEL_SHA='a'*64

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def extracted(path,name,env,owner=None):
    tree0=ast.parse(path.read_text())
    if owner is not None:
        classes=[n for n in tree0.body if isinstance(n,ast.ClassDef) and n.name==owner]
        assert len(classes)==1,owner
        nodes=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name==name]
    else:
        nodes=[n for n in ast.walk(tree0) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1,name
    tree=ast.Module(body=[copy.deepcopy(nodes[0])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),env)
    return env[name],hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest()

class FixtureModel:
    def __init__(self):
        self.config=SimpleNamespace(action_horizon=32,action_dim=32)
        self.paligemma_with_expert=SimpleNamespace(
             paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
             forward=lambda **kw:(None,()))
    def _preprocess_observation(self,o,train=False):
        assert train is False
        return [],[],None,None,o.state
    def embed_prefix(self,*args):
        return torch.zeros(1,3,4),torch.ones(1,3,dtype=torch.bool),torch.zeros(1,3,dtype=torch.bool)
    def _prepare_attention_masks_4d(self,x):return x[:,None]
    def denoise_step(self,state,pad,cache,x,t):return .125*x+.001*state[:,None,:]

class Observation:
    @staticmethod
    def from_dict(inputs):return SimpleNamespace(**inputs)

def inputs(obs):
    # Deliberate fixture, NOT the actual frozen image/token normalization chain.
    return dict(state=np.r_[np.asarray(obs['state'],np.float32),np.zeros(18,np.float32)])
ENV=dict(torch=torch,np=np,jax=jax,override=override,Tensor=torch.Tensor,time=__import__('time'),
         _model=SimpleNamespace(Observation=Observation),
         make_att_2d_masks=lambda p,a:p[:,:,None]&p[:,None,:])
sample_noise,h1=extracted(SOURCES[0],'sample_noise',ENV)
sample_actions,h2=extracted(SOURCES[0],'sample_actions',ENV)
infer,h3=extracted(SOURCES[1],'infer',ENV,owner='Policy')
FixtureModel.sample_noise=sample_noise;FixtureModel.sample_actions=sample_actions

def policy(model):
    return SimpleNamespace(_is_pytorch_model=True,_pytorch_device='cpu',
         _input_transform=inputs,_output_transform=lambda out:dict(out,actions=out['actions'][:,:14].copy()),
         _sample_kwargs={},_sample_actions=model.sample_actions)

def operation(model,state=None,noise=None):
    p=policy(model);state=np.zeros(14,np.float32) if state is None else state
    def run():
        out=infer(p,dict(state=state),noise=noise)
        return dict(base=np.asarray(out['actions'],np.float32))
    return run

def seed(router,lane,n=17):
    with router.bind(lane),router.torch_scope(lane):
        np.random.seed(n);random.seed(n);torch.manual_seed(n)

def simulate_native(draws=32):
    np.random.uniform(size=draws)
    for _ in range(draws):random.random()

def rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid sampler/fork accepted')

def run():
    router=RngRouter(cuda=False);router.install();checks=0;records=[]
    try:
        discovery=Lane('discovery');seed(router,discovery);model=FixtureModel()
        with router.bind(discovery),router.torch_scope(discovery):
            source=FreshSamplerFork(router,discovery,model,device_uuid='CPU',model_sha256=MODEL_SHA)
            for q in range(3):
                source.draw(operation(model),query=q,tick=q*100)
                if q<2:
                    simulate_native()
                    source.finish_prefix(32,end_tick=(q+1)*100)
            assert len(source.notes)==3 and len(source.ledger.records)==3;checks+=1
        for branch,effects in [('Z_reference',0),('C_original',1),('Z_sham',0)]:
            ref=Lane('reference_'+branch);seed(router,ref)
            with router.bind(ref),router.torch_scope(ref):
                ref_policy=FixtureModel()
                for q in range(3):
                    operation(ref_policy)()
                    simulate_native(33 if q==2 and effects else 32)
                ref_env=environment_hash(router.capture(ref))
                reference=operation(ref_policy,np.full(14,effects*.01,np.float32))()['base']
                reference_state=full_hash(router.capture(ref))
            legacy=Lane('legacy_'+branch);seed(router,legacy)
            with router.bind(legacy),router.torch_scope(legacy):
                simulate_native();simulate_native()
                fork=source.branch(branch,router,legacy,device_uuid='CPU',model_sha256=MODEL_SHA)
                rejected(lambda:fork.install(tick=300,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                fork.remember_legacy_boundary(device_uuid='CPU',model_sha256=MODEL_SHA)
                simulate_native(33 if effects else 32)
                fork.finish_origin(32,end_tick=300,requested95_arrived=True)
                before=router.capture(legacy)
                assert environment_hash(before)==ref_env;checks+=1
                rejected(lambda:fork.install(tick=299,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                rejected(lambda:fork.install(tick=300,device_uuid='GPU-wrong',model_sha256=MODEL_SHA));checks+=1
                assert full_hash(before)==full_hash(router.capture(legacy));checks+=1
                handoff=fork.install(tick=300,device_uuid='CPU',model_sha256=MODEL_SHA)
                assert environment_hash(before)==environment_hash(router.capture(legacy));checks+=1
                actual,note=fork.draw_following(operation(model,np.full(14,effects*.01,np.float32)),
                    query=3,tick=300,device_uuid='CPU',model_sha256=MODEL_SHA)
                assert actual['base'].tobytes()==reference.tobytes();checks+=1
                assert full_hash(router.capture(legacy))==reference_state;checks+=1
                fork.finish_following(32,end_tick=400)
                rejected(lambda:fork.draw_following(operation(model),query=4,tick=400,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                rejected(lambda:fork.install(tick=400,device_uuid='CPU',model_sha256=MODEL_SHA));checks+=1
                records.append(dict(branch=branch,reference_base_exact=True,reference_all_CPU_RNG_exact=True,handoff=handoff,note=note))
        # Every negative gets a fresh discovery; never retry a consumed bad draw.
        for mode in ('explicit_noise','numpy','python','torch_before','torch_after','double_noise','wrong_shape','raises'):
            lane=Lane(mode);seed(router,lane)
            with router.bind(lane),router.torch_scope(lane):
                m=FixtureModel();stream=FreshSamplerFork(router,lane,m,device_uuid='CPU',model_sha256=MODEL_SHA)
                def bad():
                    if mode=='numpy':np.random.random()
                    if mode=='python':random.random()
                    if mode=='torch_before':torch.rand(1)
                    if mode=='double_noise':m.sample_noise((1,32,32),'cpu')
                    if mode=='wrong_shape':m.config.action_dim=14
                    if mode=='raises':raise ValueError('Intentional fixture forward failure')
                    result=operation(m,noise=np.zeros((32,32),np.float32) if mode=='explicit_noise' else None)()
                    if mode=='torch_after':torch.rand(1)
                    return result
                rejected(lambda:stream.draw(bad,query=0,tick=0))
                assert stream.poisoned and 'sample_noise' not in m.__dict__
                rejected(lambda:stream.draw(operation(m),query=0,tick=0));checks+=3
        for mode in ('censor','unknown95','unknown_torch_draw','mutated_snapshot','model_identity','foreign_process'):
            lane=Lane(mode);seed(router,lane)
            with router.bind(lane),router.torch_scope(lane):
                m=FixtureModel();stream=FreshSamplerFork(router,lane,m,device_uuid='CPU',model_sha256=MODEL_SHA)
                stream.draw(operation(m),query=0,tick=0)
                fork=stream.branch(mode,router,lane,device_uuid='CPU',model_sha256=MODEL_SHA)
                fork.remember_legacy_boundary(device_uuid='CPU',model_sha256=MODEL_SHA)
                fork.finish_origin(22 if mode=='censor' else 32,end_tick=100,
                      requested95_arrived=mode!='unknown95')
                if mode=='unknown_torch_draw':torch.rand(1)
                if mode=='mutated_snapshot':fork._snapshot['torch_cpu'][0]^=1
                if mode=='foreign_process':stream.owner_pid=-1
                before=full_hash(router.capture(lane))
                rejected(lambda:fork.install(tick=100,device_uuid='CPU',
                          model_sha256='b'*64 if mode=='model_identity' else MODEL_SHA))
                assert before==full_hash(router.capture(lane));checks+=2
        assert not torch.cuda.is_initialized();checks+=1
    finally:router.uninstall()
    O=T/'outputs/native_nextchunk_v1'
    report=dict(passed=True,checks=checks,actual_extracted_sampler_noise_and_infer_AST_hashes=[h1,h2,h3],
         original_method_source_sha256={str(p):sha(p) for p in SOURCES},
         real_RngRouter_source_sha256=sha(T.parents[1]/'deployment/concurrent_rng.py'),
         source_sha256={str(C/n):sha(C/n) for n in ('sampler_checks.py','sampler_fork.py','query_ledger.py')},
         branches=records,source_methods_executed_without_rewriting=True,forward_is_fixture=True,
         no_full_policy_transform_or_VLA_or_CUDA_or_simulator_equivalence_claim=True,
         NOT_GPU_READY=True,scope='Actual pinned method bodies with fixture embeddings/denoise/input-output transforms, real CPU torch and original RngRouter. Proves fork/noise/control contracts only; GPU actual frozen weights/sampler/physics/zero-shadow/reference and cleanup still required.')
    file=O/'sampler_cpu_checks.json';assert not file.exists()
    file.write_text(json.dumps(report,indent=2)+'\n')
    print('FRESH_SAMPLER_CPU_CONTRACTS_PASS',checks,'NOT_GPU_READY',flush=True)
if __name__=='__main__':run()
