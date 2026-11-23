"""Load the original strictly checked frozen policy on CPU before GPU admission."""
from contextlib import contextmanager
import functools,hashlib,inspect,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
RT=Path('/data/cj/robo_pi/250010033/qfw/robotwin')

def state_digest(model):
    import torch
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(json.dumps([key,list(value.shape),str(value.dtype)]).encode());h.update(memoryview(value.detach().contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()

def load_cpu_policy(verify_state=False):
    import torch
    from cx_tta.cache_expert import create_base_policy
    from startup_loading import skip_overwritten_random_init
    proof=json.loads((D/'reports/target_fast_load_cpu_20261022.json').read_text())
    if not proof['passed']or proof['loading_helper_sha256']!=hashlib.sha256((D/'startup_loading.py').read_bytes()).hexdigest():raise RuntimeError('Frozen loading CPU proof stale')
    sys.path.insert(0,str(RT/'RoboTwin/policy/pi05/src'));sys.path.insert(0,str(RT/'pi05_integration'))
    from openpi.models.model import BaseModelConfig
    from openpi.policies.policy import Policy
    original_load=BaseModelConfig.load_pytorch;original_init=Policy.__init__;raw=[]
    @functools.wraps(original_load)
    def load(self,*args,**kwargs):
        with skip_overwritten_random_init():model=original_load(self,*args,**kwargs)
        if verify_state:
            digest=state_digest(model)
            if digest!=proof['comparisons'][0]['state_sha256']:raise RuntimeError('Original raw frozen model state differs')
            raw.append(digest)
        return model
    @functools.wraps(original_init)
    def initialize(self,*args,**kwargs):
        if kwargs.get('pytorch_device')!='cuda'or kwargs.get('is_pytorch')is not True:raise RuntimeError('Unexpected original frozen policy constructor')
        kwargs['pytorch_device']='cpu';return original_init(self,*args,**kwargs)
    BaseModelConfig.load_pytorch=load;Policy.__init__=initialize
    try:policy=create_base_policy()
    finally:BaseModelConfig.load_pytorch=original_load;Policy.__init__=original_init
    if torch.cuda.is_initialized()or policy._pytorch_device!='cpu'or any(v.device.type!='cpu'for v in policy._model.state_dict().values()):raise RuntimeError('CPU policy preparation touched GPU')
    if not hasattr(policy._sample_actions,'_torchdynamo_orig_callable'):raise RuntimeError('Original compiled sampler lost')
    return policy,dict(raw_state_sha256=raw,original_precision_conversion=True,original_transforms=True,compiled_max_autotune=True)

def move_to_gpu(policy):
    if policy._pytorch_device!='cpu':raise RuntimeError('Prepared policy was already moved')
    policy._model=policy._model.to('cuda');policy._model.eval();policy._pytorch_device='cuda';policy._sample_actions=policy._model.sample_actions
    return policy
