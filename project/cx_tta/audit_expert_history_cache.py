"""Independent readback of actual S2m arrays and raw expert physics times."""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .audit import digest
from .paths import output_path, write_json
from .train_structured_action import read, verify


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def audit(run):
    q=read(run/'training_protocol.json');verify(q['source_sha256'])
    parent=Path(q['parent_run']);p=read(parent/'protocol.json')
    mapping=read(run/'history_mapping.json')['pairs']
    a=arrays(parent/'inputs.npz');c=arrays(run/'S0_control_inputs.npz');t=arrays(run/'S_expert_history_inputs.npz')
    allowed={'visual','text','state','past_visual','past_text','past_state','has_history','ee_pose','past_ee_pose'}
    assert set(a)==set(c)==set(t)==allowed
    assert len(p['rows'])==611 and all(a[k].tobytes()==c[k].tobytes() for k in a)
    selected={r['training_row'] for r in mapping}
    assert len(selected)==len(mapping)==444
    untouched=[i for i in range(611) if i not in selected]
    for key in ('visual','text','state','ee_pose'):
        assert a[key].tobytes()==t[key].tobytes()
    for key in a:
        assert a[key][untouched].tobytes()==t[key][untouched].tobytes()
    tasks=Counter();sources={}
    for pair in mapping:
        i,j=pair['training_row'],pair['past_training_row'];current,old=p['rows'][i],p['rows'][j]
        for field in ('task','group','instruction'):
            assert current[field]==old[field]
        assert current['kind']==old['kind']=='expert'
        assert current['source_query']['path']==old['source_query']['path']==pair['source_path']
        assert current['frame']-old['frame']==32 and i!=j
        assert not 100380<=int(current['group'].rsplit(':',1)[1])<=100399
        assert a['has_history'][i,0]==0 and t['has_history'][i,0]==1
        for field in ('visual','text','state','ee_pose'):
            assert t['past_'+field][i].tobytes()==a[field][j].tobytes()
        tasks[current['task']]+=1
        path=pair['source_path']
        if path not in sources:
            assert digest(Path(path))==pair['source_sha256']
            with h5py.File(path,'r') as h:
                ticks=h['cx_measurement/physics_tick'][:].reshape(-1).astype(np.int64)
                dense=h['cx_dense/tick_before'][:].reshape(-1).astype(np.int64)
                dt=h['cx_dense/dt_s'][:].reshape(-1).astype(np.float64)
            sources[path]=(ticks,dense,np.concatenate(([0.],np.cumsum(dt))))
        ticks,dense,cumulative=sources[path];f,g=current['frame'],old['frame']
        assert int(ticks[f]-ticks[g])==pair['physics_tick_gap']
        elapsed=float(cumulative[ticks[f]-dense[0]]-cumulative[ticks[g]-dense[0]])
        assert elapsed==pair['physics_elapsed_s'] and elapsed>0
    assert len(sources)==32 and int(t['has_history'].sum())==471
    assert q['labels']==str(parent/'labels.npz')
    assert digest(parent/'labels.npz')==q['source_sha256'][str(parent/'labels.npz')]
    assert q['initial']==str(parent/'initial_seed0.pt') and len(p['development_rows'])==4
    result=dict(passed=True,rows=611,changed_expert_rows=444,changed_by_task=dict(tasks),unchanged_rows=167,
        current_fields_byte_exact=True,all_control_fields_byte_exact=True,labels_and_weights_same_file=True,
        original_history_rows=27,treatment_history_rows=471,expert_raw_sources_sha_verified=32,
        independently_verified_physics_time_pairs=len(mapping),history_domain='expert_saved_sequence',
        certified_native_query_history=False,development_cup_rows_excluded=4,GPU_used=False,
        inherited_missing_raw_sources=q['inherited_missing_sources']['missing_count'])
    assert result==read(run/'independent_cache_audit.json')
    files=[Path(__file__),run/'training_protocol.json',run/'history_mapping.json',
           run/'independent_cache_audit.json',parent/'inputs.npz',parent/'labels.npz',
           run/'S0_control_inputs.npz',run/'S_expert_history_inputs.npz']
    write_json(run/'independent_cache_reproduction.json',dict(passed=True,report_reproduced_exactly=True,
        source_sha256={str(p):digest(p) for p in files},report=result))
    print('INDEPENDENT_CACHE_AUDIT_REPRODUCED',flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();audit(output_path(args.run))


if __name__=='__main__':
    main()
