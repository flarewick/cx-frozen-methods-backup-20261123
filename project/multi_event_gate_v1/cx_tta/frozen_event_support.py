"""Explicit checkpoint registry for frozen M_event_full; no learned selector/TTA."""
import ast
from functools import lru_cache
from pathlib import Path

from .paths import ROOT, WORKSPACE_ROOT

LEGACY_TASKS = ('place_empty_cup', 'beat_block_hammer', 'place_shoe', 'turn_switch')
STACKING_TASKS = ('stack_bowls_two', 'stack_blocks_two')
REGISTRY = {
    'legacy4': dict(run=ROOT/'outputs/effect_learning/output_support_20261005_082307_793921',
                    filename='A_bounded_clipped_seed0.pt', tasks=LEGACY_TASKS,
                    history='current_and_previous_native_chunk', stage='expert_and_recovery'),
    'stacking2': dict(run=WORKSPACE_ROOT/'outputs/stacking_adapter/shared_20261014_031147_697383',
                      filename='adapter_seed0.pt', tasks=STACKING_TASKS,
                      history='current_only_adapter', stage='expert_initialization_only'),
}


def resolve_adapter(task, choice):
    if choice == 'auto':
        if task in LEGACY_TASKS: return 'legacy4'
        if task in STACKING_TASKS: return 'stacking2'
        raise ValueError('Unseen task: explicitly choose --adapter legacy4 or stacking2 for frozen transfer')
    if choice not in REGISTRY: raise ValueError('Unknown adapter')
    return choice


@lru_cache(maxsize=2)
def audited_adapter(name):
    """Read and validate actual frozen weights, including existing training audits."""
    import json
    r = REGISTRY[name]
    checkpoint = r['run']/r['filename']
    if name == 'stacking2':
        from .stacking_adapter import read_checkpoint
        payload = read_checkpoint(checkpoint)
        plan_name = 'training_plan.json'
        extra = [r['run']/'cache_manifest.json']
    else:
        from .train_output_support import read_checkpoint, BASE
        payload = read_checkpoint(checkpoint, 'A_bounded_clipped')
        plan_name = 'training_protocol.json'
        extra = [BASE/'S0_seed0.pt', BASE/'training_protocol.json', BASE/'training_audit.json']
    plan = json.loads((r['run']/plan_name).read_text())
    files = [checkpoint, r['run']/plan_name, r['run']/'training_result.json', r['run']/'training_audit.json', *extra]
    return payload['metadata'], plan, files


def deployed_sources():
    """Pin this evaluator's static local import closure, not unrelated experiments."""
    folder = WORKSPACE_ROOT/'cx_tta'
    pending = ['evaluate_frozen_event', 'frozen_event_policy', 'multi_event_policy', 'stacking_policy']
    seen = set()
    while pending:
        name = pending.pop()
        path = folder/(name+'.py')
        if name in seen or not path.is_file(): continue
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                pending.append(node.module.split('.')[0])
    return [folder/(name+'.py') for name in sorted(seen)]


def verify_base(adapter, tokenizer):
    if adapter == 'stacking2':
        from .stacking_training import verify_base as verify
        from .evaluate_stacking import read
        return verify(read(REGISTRY[adapter]['run']/'training_plan.json'))
    from .evaluate_structured_action import check_training_policy_identity
    from .train_output_support import BASE
    return check_training_policy_identity(BASE, tokenizer)


def make_policy(base, adapter, condition, config):
    run = REGISTRY[adapter]['run']
    if adapter == 'stacking2':
        from .stacking_policy import StackingPolicy
        return StackingPolicy(base, run, condition, config)
    if condition == 'M_event_full':
        from .multi_event_policy import MultiEventPolicy
        return MultiEventPolicy(base, run, condition, gate_run=None, config=config)
    if condition == 'Z_zero':
        from .frozen_event_policy import LegacyZeroPolicy
        return LegacyZeroPolicy(base, run, config)
    if condition == 'L0':
        from .official_onset_domain import OfficialOutputSupportPolicy
        return OfficialOutputSupportPolicy(base, run, 'L0', force_zero=True)
    raise ValueError('Unsupported condition')
