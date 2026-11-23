"""All4587 stream rows; verify deployed numeric API, not a new model fit."""
from pathlib import Path
import hashlib,json
import numpy as np
from native_delayed_query_v1.predictor import FrozenPredictor
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_delayed_query_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    proof=json.loads((O/'stream_parity.json').read_text())
    assert proof['passed']and sha(O/'stream_rows.npz')==proof['stream_arrays_sha256']
    with np.load(O/'stream_rows.npz',allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
    model=FrozenPredictor();checks=0;predictions=[]
    for first in range(0,len(a['target']),64):
        batch=[{k:a[k][i]for k in('context','command','start_points')}for i in range(first,min(first+64,len(a['target'])))]
        actual=model.predict(batch)
        x,u,p=(np.stack([r[k]for r in batch])for k in('context','command','start_points'))
        xx,v,kind,_=model.model.coordinates(x,u,p,'correct')
        reference=model.model.kernel(model.state,xx,v,kind)@model.state['weights']
        assert np.array_equal(actual,reference)and np.isfinite(actual).all();checks+=1
        poisoned=[dict(r,target=np.full(2,np.nan),relative_target=np.full(2,np.nan),task='must_not_read',seed=-999,success=True,physics='must_not_read')for r in batch]
        assert np.array_equal(model.predict(poisoned),actual);checks+=1
        predictions.append(actual)
    r={k:a[k][0].copy()for k in('context','command','start_points')}
    r['context'][34:44]=0;r['command'][:192]=0
    r['command'][[32,33,36,37]]=np.clip(r['context'][[6,13,6,13]]*np.pi,0,1)/.1
    r['command'][38]=3.;r['command'][39]=7.
    assert np.array_equal(model.predict([r]),np.zeros((1,2)));checks+=1
    result=np.concatenate(predictions);file=O/'stream_predictions.npz'
    with file.open('xb')as f:np.savez_compressed(f,prediction=result)
    evidence=dict(passed=True,checks=checks,rows=len(result),fixed_checkpoint_sha256=model.protocol['checkpoint_sha256'],
        no_refit_or_model_selection=True,labels_and_task_seed_outcome_poison_ignored=True,exact_physical_null=True,
        predictions_sha256=sha(file),source_sha256={str(p):sha(p)for p in(C/'predictor.py',C/'predictor_parity.py')},
        input_sha256={str(p):sha(p)for p in(O/'stream_rows.npz',O/'stream_parity.json',T/'outputs/native_phase_feedback_v1/frozen_auxiliary/protocol.json')},
        scope='Deployment numeric interface only on all old developer stream rows. Not independent performance or policy adaptation proof.',
        shared_fast128_integrated=False,online_updates=0,TTA_utility_proven=False)
    (O/'predictor_parity.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('FROZEN_AUXILIARY_DEPLOYMENT_ALL_STREAM_ROWS_PASS',checks,len(result),flush=True)
if __name__=='__main__':run()
