"""Audit old observed-history versus actual actor-image argument; never edit it."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,hashlib
import numpy as np
C=Path(__file__).resolve().parent;T=C.parent
R=T/'outputs/native_stage_v1/collections/5d5369942959181f58a604c4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run():
    source=T/'native_stage_v1/cx_tta_stage'
    tree=ast.parse((source/'parallel_runtime.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='call' and len(n.args)>7 and isinstance(n.args[1],ast.Name) and n.args[1].id=='query']
    assert len(calls)==1 and ast.unparse(calls[0].args[7])=='discovery.anchor[0]'
    query=ast.parse((source/'query.py').read_text())
    assert "zip(('cam_high', 'cam_right_wrist', 'cam_left_wrist'), images)" in ast.unparse(query)
    rows=[];inputs={}
    for casefile in sorted((R/'sessions').rglob('case.json')):
        case=read(casefile);folder=casefile.parent
        if case.get('role')!='development' or (folder/'unavailable.json').exists():continue
        p=folder/'observed_history.npz';inputs[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            cams=[z['cam'+str(c)].copy() for c in range(3)]
            n=len(z['queries'])
        changes=[sum(not np.array_equal(cams[c][q],cams[c][0]) for c in range(3)) for q in range(n)]
        rows.append(dict(case_id=case['case_id'],queries=n,
              current_observation_differs_from_fixed_actor_anchor_by_camera=changes,
              final_query_has_different_RGB_from_actor_anchor=bool(changes[-1]),
              actor_uses_initial_anchor_for_all_discovery_queries=True,
              history_RGB_is_observer_current_not_actor_image_proof=True))
    assert len(rows)==29
    report=dict(complete=True,source_cases=29,queries=sum(r['queries'] for r in rows),
        later_queries=sum(r['queries']-1 for r in rows),
        differing_later_queries=sum(sum(v>0 for v in r['current_observation_differs_from_fixed_actor_anchor_by_camera'][1:]) for r in rows),
        final_query_RGB_different_cases=sum(r['final_query_has_different_RGB_from_actor_anchor'] for r in rows),
        rows=rows,source_sha256={str(p):sha(p) for p in [Path(__file__),*[source/n for n in ('parallel_runtime.py','query.py','runtime.py','history.py','views.py','rgb.py')]]},
        input_sha256=inputs,
        clarification='Legacy controlled-view discovery actually uses fixed initial anchorRGB for base.infer even though history records current observerRGB. Historical history.semantics claiming each current observation already passed to policy is too broad for q>0. All old artifacts remain immutable; their RGB/command/physics diagnostics and old negative models are retained, never relabelled full native closed-loop policy evidence.',
        next='New natural native path MUST pass current RGB to every query and record exact actor-input hashes. Old first500/full32 equality remains separately excluded legacy compatibility, not a false equality requirement on new currentRGB trajectories. New per-query full sequential reference and RNG controls required.',
        GT_forward_or_training_allowed=False,TTA_utility_proven=False)
    out=T/'outputs/native_nextchunk_v1/actor_input_audit.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('LEGACY_ACTOR_IMAGE_AUDIT',report['source_cases'],report['queries'],
          report['later_queries'],report['differing_later_queries'],report['final_query_RGB_different_cases'],flush=True)
if __name__=='__main__':run()
