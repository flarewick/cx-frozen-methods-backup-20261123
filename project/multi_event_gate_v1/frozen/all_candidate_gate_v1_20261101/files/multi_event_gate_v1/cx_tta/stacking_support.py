"""CPU-only task/asset contract; procedural blocks need no external mesh asset."""
import json
from pathlib import Path
from .audit import digest


def task_contract(runtime, task):
    import yaml
    runtime = Path(runtime)
    task_source = runtime/'envs'/f'{task}.py'
    files = [task_source, runtime/'description/task_instruction'/f'{task}.json',
             runtime/'task_config/_eval_step_limit.yml', runtime/'task_config/demo_randomized.yml']
    if task in ('stack_bowls_two', 'lift_pot'):
        from .evaluate_onset_transfer import task_files
        extra, assets, budget = task_files(runtime, task)
        files = list(dict.fromkeys(files + extra))
    else:
        assets = []
        budget = yaml.safe_load(files[2].read_text()).get(task)
    if type(budget) is not int or budget <= 0:
        raise ValueError('Missing official native action budget')
    if task == 'stack_blocks_two':
        # create_box builds geometry and functional points; audit its implementation.
        files += [runtime/'envs/utils/create_actor.py', runtime/'envs/utils/actor_utils.py']
    for f in files + assets:
        if not f.is_file(): raise FileNotFoundError(f)
    language = json.loads(files[1].read_text())
    if not all(language.get(k) for k in ('seen', 'unseen')):
        raise ValueError('Both official language splits are required')
    provenance = json.loads((runtime.parent/'provenance.json').read_text())['runtime_sha256']
    for f in files:
        if f.suffix == '.py' and digest(f) != provenance[str(f.relative_to(runtime))]:
            raise ValueError('Task implementation differs from frozen runtime: '+str(f))
    return files, assets, budget
