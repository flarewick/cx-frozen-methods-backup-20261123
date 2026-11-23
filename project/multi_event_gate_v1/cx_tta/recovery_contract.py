"""Pure contracts for paired terminal continuations; no simulator imports."""
import copy
import hashlib
import json
import random
import numpy as np


def rng_capture():
    import torch
    return dict(python=copy.deepcopy(random.getstate()),numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_restore(state):
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def rng_fingerprint(state):
    h=hashlib.sha256();h.update(repr(state['python']).encode())
    name,keys,pos,gauss,value=state['numpy']
    h.update(json.dumps([name,pos,gauss,value]).encode());h.update(keys.tobytes())
    for value in [state['torch_cpu'],*state['torch_cuda']]:h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def terminal_label(count,limit,success,*,interrupted=False):
    if not 0<=count<=limit or limit<=0:raise ValueError('Invalid native action budget')
    if interrupted:return dict(complete=False,success=None,reason='interrupted')
    if success:return dict(complete=True,success=True,reason='native_success')
    if count==limit:return dict(complete=True,success=False,reason='native_budget_exhausted')
    return dict(complete=False,success=None,reason='unfinished')


def execute_chunk(task,actions,after_action=None):
    """Same 32 callback attempts/get_obs as official; no budget reset or early break."""
    actions=np.asarray(actions)
    if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError('Expected finite native chunk')
    start=int(task.take_action_cnt);observation=None
    for i,action in enumerate(actions):
        before=int(task.take_action_cnt);terminal=bool(task.eval_success) or before>=task.step_lim
        task.take_action(action);observation=task.get_obs();after=int(task.take_action_cnt)
        if after-before!=(0 if terminal else 1):raise RuntimeError('Unexpected action advancement')
        if not 0<=after<=task.step_lim:raise RuntimeError('Native budget exceeded')
        if after_action is not None:after_action(i,action,observation,after-before)
    return observation,int(task.take_action_cnt)-start


def comparison(base,candidate,sham_passed):
    if not sham_passed:return 'invalid_sham'
    if not base['complete'] or not candidate['complete']:return 'censored'
    if candidate['success']==base['success']:return 'tie'
    return 'rescued' if candidate['success'] else 'harmed'
