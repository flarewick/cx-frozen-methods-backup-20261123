"""Freeze bounded semantic/behavior-preservation pilot from existing training data."""
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import ROOT, new_run, write_json

DIRECTION = ROOT/'outputs/effect_learning/expert_direction_supervision_20260930_133659_564181'
PRIOR = ROOT/'outputs/effect_learning/direction_adapter_pilot_20260930_134247_051603'


def rank(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    direction=json.loads((DIRECTION/'index.json').read_text());prior=json.loads((PRIOR/'protocol.json').read_text())
    groups={part:{r['group'] for r in rows} for part,rows in prior['rows'].items()}
    records=[]
    for task in sorted({r['task'] for r in direction['records']}):
        for part,count in [('fit',12),('inner_held',3)]:
            eligible=[r for r in direction['records'] if r['status']=='prepared' and r['task']==task and r['group'] in groups[part]]
            chosen=sorted(eligible,key=lambda r:rank(r['group']))[:count]
            if len(chosen)!=count:raise ValueError('Missing frozen episode subset')
            for row in chosen:records.append(dict(row,pilot_split=part,kind='expert'))
    plan=json.loads((ROOT/'configs/intervention_batch1_20260927.json').read_text())
    allowed={(s['task'],s['seed']) for s in plan['scenes'] if s['split']=='train' and s['source_l0_success']}
    held=set()
    for task in sorted({t for t,s in allowed}):
        seeds=sorted([s for t,s in allowed if t==task])
        if len(seeds)>=2:held.add((task,seeds[-1]))
    anchors={};source_cache={};rejected=[]
    for family in ('cartesian_pulse_batches','short_window_batches','intervention_batches'):
        for ip in sorted((ROOT/'outputs'/family).glob('*/*/index.json')):
            index=json.loads(ip.read_text())
            if index['status']!='complete':raise ValueError('Incomplete source index')
            for job in index['records']:
                scene=(job['task'],job['seed']);key=(*scene,job['chunk'])
                if scene not in allowed or key in anchors:continue
                run=Path(job['run']);m=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
                if result['status']!='complete' or not result['source_observation_exact_match']:raise ValueError('Unverified replay')
                source=Path(m['source']);trace=source/'traces'/f'r0_seed{job["seed"]}.jsonl'
                if digest(trace)!=m['source_trace_sha256'] or digest(source/'manifest.json')!=m['source_manifest_sha256']:raise ValueError('Changed L0 source')
                if (source,job['seed']) not in source_cache:
                    sm=json.loads((source/'manifest.json').read_text());ep=next(r for r in json.loads((source/'episodes.json').read_text()) if r['seed']==job['seed'])
                    if not ep['success'] or not ep['eligible'] or sm['args']['condition']!='L0' or sm['args']['rollout_protocol']!='official':raise ValueError('Invalid successful train L0')
                    source_cache[source,job['seed']]=(ep,[json.loads(x) for x in trace.read_text().splitlines()])
                ep,steps=source_cache[source,job['seed']];step=steps[job['chunk']]
                checkpoint=run/'checkpoint.npz'
                with np.load(checkpoint) as z:
                    base=z['base_actions'];state=z['command_state'];images=[z[k+'_rgb'] for k in ('head','right','left')]
                if step['executed']!=32 or not np.array_equal(base,np.asarray(step['base_executed'],np.float32)) or array_hash(base)!=step['base_chunk_sha256']:raise ValueError('Wrong chunk action')
                if any(array_hash(im)!=step['observation_components']['images'][camera] for im,camera in zip(images,('head_camera','right_camera','left_camera'))):raise ValueError('Wrong decision image')
                cmp=json.loads((run/'observation_comparison.json').read_text())
                if not cmp['exact_match'] or cmp['source']!=step['observation_components']:raise ValueError('Wrong decision signature')
                frame=next((f for f in result['branches']['base'].get('frames',[]) if f['offset']==0),None)
                if frame is not None:
                    with np.load(run/frame['observed_file']) as z:
                        if array_hash(z['command_state_raw'])!=step['observation_components']['command_state'] or not np.array_equal(state,z['command_state_raw'].astype(np.float32)):raise ValueError('Wrong raw command state')
                elif not any(array_hash(candidate)==step['observation_components']['command_state'] for candidate in (state,state.astype(np.float64))):
                    rejected.append(dict(run=str(run),reason='No raw command state matching source hash'));continue
                anchors[key]=dict(kind='anchor',task=scene[0],seed=scene[1],group=f'{scene[0]}:{scene[1]}',chunk=job['chunk'],pilot_split='anchor_held' if scene in held else 'anchor_fit',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),run=str(run),source=str(source),source_trace_sha256=m['source_trace_sha256'],result_sha256=digest(run/'result.json'),instruction=ep['instruction'],decision_image_hashes=step['observation_components']['images'],base_chunk_sha256=step['base_chunk_sha256'])
    if {key[:2] for key in anchors}!=allowed:raise ValueError('Missing successful scene coverage')
    records.extend(anchors[key] for key in sorted(anchors))
    run=new_run('semantic_pilot_plan',group=('effect_learning',))
    write_json(run/'plan.json',dict(status='frozen',rejected=rejected,records=records,direction_dataset=str(DIRECTION),direction_index_sha256=digest(DIRECTION/'index.json'),prior_protocol_sha256=digest(PRIOR/'protocol.json'),scope='Existing data only; 12 fit and 3 inner-held expert episodes/task selected by group hash within prior frozen split, no outcomes used for expert selection. Current decision RGB+instruction+command only. Successful L0 is used solely for offline preservation regularization, NOT an optimal-action or recovery label. Original 40 expert validation and 6 intervention validation scenes excluded.',anchor_scope='7 success train scenes; last seed of multi-scene tasks held: cup/shoe only. No held success scene available for hammer/switch; do not claim their independent preservation.',feature_contract='Frozen checkpoint VLM prefix joint representation; no action denoising, noise sampling, teacher future or GT. Deterministic eager feature extraction is auxiliary; official compiled policy protocol unchanged.',training_contract=dict(seed=0,epochs=10,lr=.0003,auxiliary_weight=.1,preservation_weight=1.,pca_components=64,modes=['coarse_bc','semantic_bc','semantic_direction','semantic_bc_preserve','semantic_direction_preserve'],history='Previous available cached observation from SAME episode with strictly smaller frame/chunk; no future. Sampling gaps may differ by source. PCA/scaler fit on expert fit only.')))
    print(run/'plan.json',flush=True)
    print(dict(Counter(r['pilot_split'] for r in records)),flush=True)

if __name__=='__main__':main()
