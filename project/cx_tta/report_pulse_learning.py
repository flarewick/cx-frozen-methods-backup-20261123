"""Report grouped effect errors and threshold diagnostics without selecting directions."""
import argparse,json
from pathlib import Path
import numpy as np
from .paths import new_run,write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();m=json.loads((a.run/'manifest.json').read_text());dataset=Path(m['dataset']);index=json.loads((dataset/'index.json').read_text());records=index['records']
    with np.load(dataset/'labels_offline.npz') as z:scale=z['scale']
    report=[]
    for unit in ('group','task'):
        with np.load(a.run/(unit+'_predictions.npz')) as z:data=dict(z)
        for task,entity,horizon in sorted({(r['task'],r['entity'],r['horizon']) for r in records}):
            ids=np.array([i for i,r in enumerate(records) if (r['task'],r['entity'],r['horizon'])==(task,entity,horizon)])
            for kind,cols,threshold in [('translation_m',np.arange(3),.005),('rotation_rad',np.arange(3,6),.02),('joint_fraction',np.array([6]),.005)]:
                valid=data['mask'][np.ix_(ids,cols)].all(1)&data['supported'][np.ix_(ids,cols)].all(1);take=ids[valid]
                if not len(take):continue
                y=data['target'][np.ix_(take,cols)]*scale[cols];norm=np.linalg.norm(y,axis=1);events=norm>threshold;near_zero=norm<threshold/10;modes={}
                for mode in ('conditioned','action_only','no_action','shuffled'):
                    pred=data[mode][np.ix_(take,cols)]*scale[cols];pnorm=np.linalg.norm(pred,axis=1);sq=np.mean((pred-y)**2,axis=1)
                    event_cos=np.sum(pred[events]*y[events],axis=1)/np.maximum(pnorm[events]*norm[events],1e-12)
                    modes[mode]=dict(unweighted_mse=float(sq.mean()),event_mse=float(sq[events].mean()) if events.any() else None,near_zero_mse=float(sq[near_zero].mean()) if near_zero.any() else None,event_cosine_mean=float(event_cos.mean()) if len(event_cos) else None,event_positive_dot=int((event_cos>0).sum()),predicted_above_threshold_on_below_threshold=int(((pnorm>threshold)&~events).sum()))
                report.append(dict(unit=unit,task=task,entity=entity,horizon=horizon,kind=kind,rows=len(take),event_rows=int(events.sum()),near_zero_rows=int(near_zero.sum()),event_scenes=len({records[i]['group'] for i in take[events]}),threshold=threshold,modes=modes))
    run=new_run('pulse_learning_diagnostics',group=('reports','audits'));write_json(run/'result.json',dict(status='complete',training_run=str(a.run),reports=report,scope='Unweighted threshold diagnostics complement scene-balanced primary MSE. All directions retained; no threshold filtering during training. Near-zero means norm below one tenth of pre-existing event threshold. Not task success.'));print(run/'result.json')

if __name__=='__main__':main()
