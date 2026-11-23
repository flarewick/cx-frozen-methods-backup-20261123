"""Immutable task sets, source contracts and disjoint scene splits."""
import ast
import hashlib
import json
import re
from pathlib import Path

from cx_tta.paths import WORKSPACE_ROOT, ROOT, output_path, write_json
from cx_tta.audit import digest
from cx_tta.collection_jobs import read

HERE = Path(__file__).resolve().parents[1]
OUTPUT = WORKSPACE_ROOT / 'outputs/task_adapters'
TEMPLATE = WORKSPACE_ROOT / 'outputs/stacking_adapter/shared_20261014_031147_697383/training_plan.json'
RUNTIME = ROOT.parent.parent / 'qfw/robotwin/RoboTwin'
SIX = ('lift_pot', 'open_laptop', 'adjust_bottle', 'grab_roller', 'dump_bin_bigbin', 'handover_mic')
SCOPE = 'task_shared_expert_initialization_v1'


def tasks_arg(value):
    tasks = sorted(set(value.split(',')))
    if not tasks or any(not re.fullmatch(r'[a-z][a-z0-9_]*', t) for t in tasks):
        raise ValueError('Use official comma-separated task names')
    return tasks


def splits(fit=40, val=10):
    if not 1 <= fit <= 1000 or not 1 <= val <= 1000:
        raise ValueError('Require 1..1000 demonstrations per split')
    return dict(smoke=dict(start=7299000, stop=7299100, episodes=1, role='interface_only'),
                adapter_fit=dict(start=7300000, stop=7301000, episodes=fit, role='independent_train'),
                adapter_validation=dict(start=7310000, stop=7311000, episodes=val, role='development_holdout'))


def local_sources():
    """Pin the static local import closure, including reused collection/training code."""
    files = list((HERE/'task_adapters').glob('*.py')) + list((HERE/'scripts').glob('*.sh'))
    pending = ['collect_stacking_resumable', 'collect_stacking_experts_v2', 'stacking_recording',
               'stacking_training_data', 'cache_expert', 'cache_joint_prefix', 'structured_features',
               'structured_action_adapter', 'stacking_adapter', 'native_entry_storage', 'stacking_training',
               'stacking_policy', 'evaluate_stacking_adapter', 'evaluation_resume']
    seen = set()
    while pending:
        name = pending.pop(); path = WORKSPACE_ROOT/'cx_tta'/(name+'.py')
        if name in seen or not path.is_file(): continue
        seen.add(name); files.append(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                pending.append(node.module.split('.')[0])
    return sorted(set(files))


def task_contract(runtime, task, config):
    import yaml
    if not re.fullmatch(r'[a-z][a-z0-9_]*', task): raise ValueError('Invalid task name')
    files = [runtime/'envs'/(task+'.py'), runtime/'description/task_instruction'/(task+'.json'),
             runtime/'task_config'/(config+'.yml'), runtime/'task_config/_eval_step_limit.yml']
    for f in files:
        if not f.is_file(): raise FileNotFoundError(f)
    language = read(files[1]); budget = yaml.safe_load(files[3].read_text()).get(task)
    if not all(language.get(k) for k in ('seen','unseen')) or type(budget) is not int or budget < 1:
        raise ValueError('Missing language or official action budget: '+task)
    # This is a static check only; dynamic model IDs/planning require the real smoke.
    tree = ast.parse(files[0].read_text())
    models = sorted({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                     and isinstance(n.value,str) and re.fullmatch(r'\d{3}_[a-zA-Z0-9_]+',n.value)})
    for model in models:
        folder = runtime/'assets/objects'/model
        if not folder.is_dir(): raise FileNotFoundError(folder)
    return dict(task=task, native_budget=budget, asset_families=models,
                source_sha256={str(f):digest(f) for f in files}, real_smoke_passed=False)


def verify(p):
    if p.get('schema') != SCOPE: raise ValueError('Wrong protocol schema')
    for f,sha in p['source_sha256'].items():
        if digest(Path(f)) != sha: raise ValueError('Frozen dependency changed: '+f)
    return p


def prepare(tasks, task_config='demo_randomized', fit=40, val=10, updates=1400):
    if task_config not in ('demo_randomized','demo_clean') or not 1 <= updates <= 100000:
        raise ValueError('Invalid environment/budget')
    template = read(TEMPLATE)
    if digest(Path(template['initial']))!=template['initial_sha256']:
        raise ValueError('Initialization checkpoint changed')
    from cx_tta.evaluate_structured_action import base_identity
    base=base_identity(Path(template['tokenizer']))
    for key in ('checkpoint','base_checkpoint_sha256','normalization_sha256','tokenizer_sha256'):
        if base[key]!=template['policy_identity'][key]:raise ValueError('Frozen VLA differs: '+key)
    checks = [task_contract(RUNTIME,t,task_config) for t in tasks]
    files = local_sources() + [TEMPLATE]
    for name,sha in template['source_sha256'].items():
        f=Path(name)
        if f.suffix=='.urdf' or '/qfw/robotwin/' in name and f.suffix=='.py':
            if digest(f)!=sha:raise ValueError('Frozen VLA/robot code changed: '+name)
            files.append(f)
    for field in ('initial','projection','robot_config','tokenizer'):
        files.append(Path(template[field]))
    # Original simulator source, language/configs are frozen for every stage.
    files += [f for part in ('envs','task_config','description') for f in (RUNTIME/part).rglob('*')
              if f.is_file() and f.suffix in ('.py','.yml','.yaml','.json')]
    p = dict(schema=SCOPE,tasks=tasks,task_config=task_config,splits=splits(fit,val),
             updates=updates,runtime_source=str(RUNTIME),tokenizer=template['tokenizer'],
             source_sha256={str(f):digest(f) for f in sorted(set(files))},task_checks=checks,
             shared=True,video=False,gate_trained=False,TTA=False,
             targets='Expert saved future commands; initialization only, not native recovery labels',
             evaluation='Official seed=0 starts at scene100000; never train on evaluation outcomes',
             initialization='Same frozen legacy4 checkpoint/PCA as stacking2; no holdout PCA fit')
    key=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:16]
    label=tasks[0] if len(tasks)==1 else 'shared'+str(len(tasks))
    run=output_path(OUTPUT/(label+'_'+key)); run.mkdir(parents=True,exist_ok=True)
    path=run/'protocol.json'
    if path.exists():
        if read(path)!=p: raise ValueError('Existing protocol differs')
    else: write_json(path,p)
    return run, verify(p)


def load(run):
    run=output_path(Path(run).resolve())
    if not run.is_relative_to(OUTPUT): raise ValueError('Not a task-adapter dataset')
    return verify(read(run/'protocol.json'))
