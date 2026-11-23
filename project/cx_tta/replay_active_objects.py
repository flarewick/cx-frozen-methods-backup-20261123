"""Narrow correction for a proven stale diagnostic alias; never alter physics."""
import copy,json
from pathlib import Path
from .audit import digest
from .probe_action_replay import compare_physical


def compare_source_physical(source, seed, reference, task, actual, tolerance=1e-5):
    raw=compare_physical(reference,actual,tolerance)
    if raw['matched']:return dict(**raw,raw_comparison=raw,diagnostic_alias_correction=None)
    if raw.get('reason')!='entity_set_changed' or raw.get('missing')!=['object_poses/cluttered_obj'] or raw.get('extra'):
        return raw
    # No created clutter in this fresh scene. A live actor must never be dropped.
    if hasattr(task,'cluttered_obj') or task.record_cluttered_objects or task.cluttered_objs:
        return raw
    source=Path(source);manifest=json.loads((source/'manifest.json').read_text());seeds=manifest['seeds'];index=seeds.index(seed)
    if index<1:return raw
    prior=source/'traces'/f'r0_seed{seeds[index-1]}.jsonl'
    if not prior.exists():return raw
    previous=[json.loads(x) for x in prior.read_text().splitlines()]
    current_file=source/'traces'/f'r0_seed{seed}.jsonl';current=[json.loads(x) for x in current_file.read_text().splitlines()]
    alias=reference['offline_only']['object_poses']['cluttered_obj']
    if not previous or previous[-1]['after']['offline_only']['object_poses'].get('cluttered_obj')!=alias:return raw
    if any(row[when]['offline_only']['object_poses'].get('cluttered_obj')!=alias for row in current for when in ('before','after')):return raw
    cleaned=copy.deepcopy(reference);del cleaned['offline_only']['object_poses']['cluttered_obj']
    checked=compare_physical(cleaned,actual,tolerance)
    return dict(**checked,raw_comparison=raw,diagnostic_alias_correction=dict(
        alias='cluttered_obj',pose=alias,prior_seed=seeds[index-1],prior_trace_sha256=digest(prior),
        source_trace_sha256=digest(current_file),constant_entire_source=True,previous_terminal_exact=True,
        fresh_task_has_alias=False,created_clutter=[],active_actor_names=sorted(a.get_name() for a in task.scene.get_all_actors()),
        rationale='Base task resets clutter lists but clean-background early return leaves previous task-instance alias; object_poses iterates vars(task). No live simulation state removed or changed.'))
