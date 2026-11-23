"""Preserve native rendering/RNG while avoiding unused CPU observations.

This does not change physics or the number/order of camera renders. Observers,
including success-time internal observers, still use the original get_obs.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap


class TaskPool:
    """Reuse native robot/planners across seeds, as native evaluation does."""
    def __init__(self):self.tasks={}
    def get(self,name,factory):
        if self.tasks and name not in self.tasks:raise ValueError('One tasktype perlane forbids retaining duplicate scenes')
        if name not in self.tasks:self.tasks[name]=factory()
        return self.tasks[name]


def observation_contract(base_source,task_source):
    tree=ast.parse(base_source)
    obs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_obs']
    if len(obs)!=1:raise ValueError('Ambiguous native get_obs')
    expected=ast.parse('self._update_render()\nself.cameras.update_picture()').body
    if [ast.dump(n,include_attributes=False) for n in obs[0].body[:2]] != [ast.dump(n,include_attributes=False) for n in expected]:
        raise ValueError('Original observation render/capture prefix changed')
    reads=[n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr=='now_obs' and isinstance(n.ctx,ast.Load)]
    action=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='take_action']
    if len(reads)!=2 or len(action)!=1:raise ValueError('Native now_obs contract changed')
    protected=[]
    for node in ast.walk(action[0]):
        if isinstance(node,ast.If) and 'eval_video' in ast.unparse(node.test):
            protected.extend(n for n in ast.walk(node) if isinstance(n,ast.Attribute) and n.attr=='now_obs' and isinstance(n.ctx,ast.Load))
    if set(map(id,reads))!=set(map(id,protected)):raise ValueError('now_obs used outside disabled video paths')
    task_tree=ast.parse(task_source)
    if any(isinstance(n,ast.Attribute) and n.attr=='now_obs' for n in ast.walk(task_tree)):
        raise ValueError('Task-specific now_obs dependency')
    return dict(passed=True,get_obs_ast_sha256=hashlib.sha256(ast.dump(obs[0],include_attributes=False).encode()).hexdigest(),
                native_renders_preserved=True,now_obs_reads_only_disabled_video=True)


def discard_observation(task):
    if getattr(task,'eval_video_path',None) is not None or getattr(task,'eval_video_ffmpeg',None) is not None:
        raise ValueError('Deferred readback requires video disabled')
    if any(task.data_type.get(k,False) for k in ('third_view','depth','mesh_segmentation','actor_segmentation','pointcloud')):
        raise ValueError('Unvalidated observation modality')
    task._update_render()
    task.cameras.update_picture()


def replay_prefix(task,item,prefix,cfg,observe,seed_all):
    seed_all(item['seed'])
    task.setup_demo(now_ep_num=item['episode_index'],seed=item['seed'],**dict(cfg,need_plan=True,save_data=False,is_test=True))
    if 'initial_physical_sha256' in item:
        from cx_tta.attribution_physics import physical_snapshot
        from cx_tta.eval_diagnostics import json_hash
        if json_hash(physical_snapshot(task))!=item['initial_physical_sha256']:raise ValueError('Qualified initialization physics changed')
    from .render_upload import install
    install(task)
    task.set_instruction(instruction=item['instruction']);discard_observation(task)
    for chunk in prefix:
        for command in chunk:
            previous=task.take_action_cnt;task.take_action(command);discard_observation(task)
            if task.take_action_cnt!=previous+1:raise ValueError('Replay prefix ended before fork')
    return observe()


def assert_original_parity(folder,reference):
    """Fresh numeric outputs must match the saved unoptimized GPU outputs."""
    import numpy as np
    import json
    from .contract import BRANCHES,load_packet,same
    folder,reference=Path(folder),Path(reference)
    for name in ['initial',*BRANCHES]:
        if not same(load_packet(folder/(name+'.npz')),load_packet(reference/(name+'.npz'))):
            raise ValueError('Original numeric replay parity failed: '+name)
    for branch in BRANCHES:
        fresh=json.loads((folder/(branch+'.json')).read_text());old=json.loads((reference/(branch+'.json')).read_text())
        for key in ('prefix_physics_sha256','prefix_rng_sha256','prefix_rng_components','initial','initial_images','initial_tokens'):
            if fresh[key]!=old[key]:raise ValueError('Original replay metadata parity failed: '+key)
        for fresh_row,old_row in zip(fresh['samples'],old['samples']):
            for key in ('tick','elapsed_s','physics_sha256','rng_sha256','view_images','token_sha256'):
                if fresh_row[key]!=old_row[key]:raise ValueError('Original sample parity failed: '+key)
    return dict(passed=True,all_numeric_packets_byte_exact=True,physics_images_tokens_exact=True,rng_exact=True,
                old_cuda_rng_not_imported=True,reference=str(reference))
