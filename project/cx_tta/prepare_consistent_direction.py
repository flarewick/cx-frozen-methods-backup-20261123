"""Derive box-consistent expert command FK labels; no simulator or extra data."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .audit_correction_budget import CAPS
from .paths import new_run, write_json
from .prepare_direction_supervision import direction_targets
from .robot_kinematics import RobotFK
from .train_semantic_pilot import load_features


def bounded_direction(base, expert, valid, robots):
    caps = CAPS.astype(np.float32)
    command = base + np.clip(expert-base, -caps, caps)
    return direction_targets(base, command, valid, robots)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();threadpool_limits(2)
    baseline=json.loads((a.baseline/'protocol.json').read_text())
    audited=json.loads((a.baseline/'paired_audit.json').read_text())
    if audited['status']!='passed':raise ValueError('Unaudited BC baseline')
    source=Path(baseline['source_run']);previous=json.loads((source/'protocol.json').read_text())
    plan,rows,arrays,_,_=load_features(Path(previous['cache']))
    if rows!=previous['rows']:raise ValueError('Changed row order')
    with np.load(source/'semantic_preprocess.npz') as z:pre=dict(z)
    old=json.loads((a.baseline/'data_contract.json').read_text())
    if digest(source/'semantic_preprocess.npz')!=old['preprocessing_sha256']:raise ValueError('Changed preprocessing')
    visual=(arrays['joint']-pre['pca_mean'])@pre['pca_components'].T
    current=np.c_[visual,arrays['coarse'][:,:14]];past=np.zeros_like(current)
    history=pre['history_indices'];exists=history>=0;past[exists]=current[history[exists]]
    context=((np.c_[current,past,exists.astype(np.float32)]-pre['context_mean'])/pre['context_std']).astype(np.float32)
    sources={r['group']:r for r in plan['records'] if r['kind']=='expert'}
    robots={}; bounded=np.zeros_like(arrays['direction']);checks=0
    for i,r in enumerate(rows):
        if r['kind']!='expert':continue
        src=sources[r['group']]
        if src['split']!='train':raise ValueError('Original validation accessed')
        if r['group'] not in robots:
            config=Path(src['collection'])/'resolved_config.json'
            if digest(config)!=src['config_sha256']:raise ValueError('Changed config')
            robots[r['group']]=[RobotFK(json.loads(config.read_text()),side) for side in ('left','right')]
            if any(digest(robot.urdf)!=src['urdf_sha256'] for robot in robots[r['group']]):raise ValueError('Changed FK source')
        base,expert,valid=[arrays[k][i] for k in ('base','target','valid')]
        raw,mask=direction_targets(base,expert,valid,robots[r['group']])
        if not np.array_equal(mask,arrays['direction_valid'][i]) or not np.allclose(raw,arrays['direction'][i],rtol=1e-6,atol=1e-6):
            raise ValueError('Original FK targets not reproduced')
        bounded[i],mask2=bounded_direction(base,expert,valid,robots[r['group']])
        if not np.array_equal(mask,mask2):raise ValueError('Bounded label changed validity')
        checks+=1
    run=new_run('consistent_direction_data',group=('effect_learning',))
    np.savez_compressed(run/'data.npz',context=context,base=arrays['base'],target=arrays['target'],valid=arrays['valid'],
        raw_direction=arrays['direction'],bounded_direction=bounded,direction_valid=arrays['direction_valid'])
    write_json(run/'index.json',dict(status='complete',baseline=str(a.baseline.resolve()),baseline_result_sha256=digest(a.baseline/'result.json'),
        baseline_audit_sha256=digest(a.baseline/'paired_audit.json'),baseline_protocol_sha256=digest(a.baseline/'protocol.json'),
        data_sha256=digest(run/'data.npz'),rows=rows,partitions=old['partitions'],
        raw_fk_queries_reproduced=checks,code_sha256=digest(Path(__file__)),sources=list(sources.values()),
        preprocessing_sha256=old['preprocessing_sha256'],scope='Same cached 48 fit / 12 inner-held expert episodes plus prior anchors. Original validation unopened. Project native expert command into existing residual box BEFORE FK. Targets and validity are loss-only, never model inputs. Box feasibility is not physical or task feasibility.'))
    print(json.dumps(dict(output=str(run),expert_queries=checks,total_rows=len(rows))),flush=True)


if __name__=='__main__':main()
