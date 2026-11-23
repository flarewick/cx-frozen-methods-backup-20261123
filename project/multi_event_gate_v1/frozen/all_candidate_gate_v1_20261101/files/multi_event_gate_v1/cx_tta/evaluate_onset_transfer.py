"""Bowls/pot task extension of the frozen official 100-episode evaluator.

Only CLI task registration and additional preflight provenance change. Bind
options/main in a private namespace so no old file or shared module is changed.
The original execute function and all four actors are used without modification.
"""
import re
from pathlib import Path
from types import FunctionType
import xml.etree.ElementTree as ET

from . import evaluate_onset_official as core
from .audit import digest

TASKS = ('stack_bowls_two', 'lift_pot')


def task_files(runtime, task):
    import yaml
    files = [runtime/'description/task_instruction'/f'{task}.json',
             runtime/'task_config/_eval_step_limit.yml', runtime/'task_config/demo_randomized.yml']
    limits = yaml.safe_load(files[1].read_text())
    if not isinstance(limits.get(task), int) or limits[task] <= 0:
        raise ValueError('Missing official task action budget')
    if task == 'stack_bowls_two':
        files += [runtime/'description/objects_description/002_bowl/base3.json']
        assets = [runtime/'assets/objects/002_bowl'/name for name in
                  ('model_data3.json', 'collision/base3.glb', 'visual/base3.glb')]
    else:
        files += [runtime/'description/objects_description/060_kitchenpot'/f'base{i}.json' for i in (0,1)]
        folder=runtime/'assets/objects/060_kitchenpot'
        models=sorted((p for p in folder.iterdir() if p.is_dir() and p.name!='visual'),
                      key=lambda p:int(re.search(r'\d+',p.name).group()) if re.search(r'\d+',p.name) else 0)
        if len(models)<2:raise FileNotFoundError('Both official pot model variants required')
        assets=[]
        for model in models[:2]:
            urdf=model/'mobility.urdf'
            assets += [urdf,model/'model_data.json']
            assets += [model/n.attrib['filename'] for n in ET.parse(urdf).iter('mesh')]
    for f in files+assets:
        if not f.is_file():raise FileNotFoundError(f)
    for f in files:
        if f.suffix=='.json':
            values=core.read(f)
            if not all(values.get(k) for k in ('seen','unseen')):
                raise ValueError('Both instruction/description splits required: '+str(f))
    return files, assets, limits[task]


def preflight(args):
    if args.task not in TASKS:raise ValueError('This entry is reserved for bowls/pot transfer')
    p,ref,tokenizer,manifest=core.preflight(args)
    runtime=ref/'robotwin'
    extra,assets,budget=task_files(runtime,args.task)
    expected=core.read(ref/'provenance.json')['runtime_sha256']
    key='envs/'+args.task+'.py'
    if digest(runtime/key)!=expected[key]:raise ValueError('Task source differs from audited runtime')
    manifest.update(transfer_entry=True, official_task_action_budget=budget,
        adaptation_scope='No new task data/training/threshold changes; fixed four-condition one-shot comparison',
        generalization_scope='Task transfer of frozen added adapter/gates; no claim task absent from base VLA pretraining',
        asset_preflight=[str(f) for f in assets])
    manifest['source_sha256'].update({str(f):digest(f) for f in
        extra+[Path(__file__),core.ROOT/'scripts/eval_onset_transfer_100.sh']})
    return p,ref,tokenizer,manifest


_namespace=dict(vars(core), TASKS=TASKS, preflight=preflight)
options=FunctionType(core.options.__code__,_namespace,'options',core.options.__defaults__,core.options.__closure__)
_namespace['options']=options
main=FunctionType(core.main.__code__,_namespace,'main',core.main.__defaults__,core.main.__closure__)

if __name__=='__main__':main()
