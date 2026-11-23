"""Verify saved predictions and compare task-balanced native/FK outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_correction_budget import CAPS, balanced
from .paths import write_json
from .robot_kinematics import RobotFK, cartesian_residual

OFFSETS=np.array([0,3,7,15,31])


def summarize_query(error,rows):
    return {task:balanced(error[[i for i,r in enumerate(rows) if r['task']==task]],
                [r for r in rows if r['task']==task]) for task in sorted({r['task'] for r in rows})}


def geometry_error(planned,target,valid,rows):
    results={}
    for kind,cols,scale in [('translation_m2',[0,1,2,7,8,9],.01),('rotation_rad2',[3,4,5,10,11,12],.1)]:
        error=(((planned[:,:,cols]-target[:,:,cols])*scale)**2).mean(-1)
        query=(error*valid).sum(-1)/valid.sum(-1)
        results[kind]=summarize_query(query,rows)
    return results


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());result=json.loads((a.run/'result.json').read_text())
    dataset=Path(protocol['dataset']);index=json.loads((dataset/'index.json').read_text())
    if result['status']!='complete' or len(result['reports'])!=9 or digest(dataset/'index.json')!=protocol['dataset_index_sha256'] or digest(dataset/'data.npz')!=index['data_sha256']:
        raise ValueError('Incomplete or changed experiment')
    with np.load(dataset/'data.npz') as z:labels=dict(z)
    robots={}
    for source in index['sources']:
        if source['pilot_split']!='inner_held':continue
        path=Path(source['collection'])/'resolved_config.json'
        if digest(path)!=source['config_sha256']:raise ValueError('Changed kinematics')
        robots[source['group']]=[RobotFK(json.loads(path.read_text()),s) for s in ('left','right')]
        if any(digest(r.urdf)!=source['urdf_sha256'] for r in robots[source['group']]):raise ValueError('Changed URDF')
    checked=0;geometries=[]
    for report in result['reports']:
        if digest(Path(report['weights']))!=report['weights_sha256']:raise ValueError('Changed weights')
        for split,record in report['files'].items():
            path=Path(record['path'])
            if digest(path)!=record['sha256']:raise ValueError('Changed predictions')
            with np.load(path) as z:values=dict(z)
            ids=values['indices'];rows=[index['rows'][i] for i in ids]
            if ids.tolist()!=index['partitions'][split] or any(not np.array_equal(values[k],labels[k][ids]) for k in ('base','target','valid')):
                raise ValueError('Unpaired observation/target')
            base,pred,target=[values[k].astype(float) for k in ('base','predicted','target')];valid=values['valid']
            if not np.isfinite(pred).all() or (np.abs(pred-base)>CAPS+1e-6).any():raise ValueError('Invalid bounded prediction')
            feasible=base+np.clip(target-base,-CAPS,CAPS)
            for name,x,y in [('expert_mse',pred,target),('base_expert_mse',base,target),('reachable_mse',pred,feasible),('base_reachable_mse',base,feasible),('residual_energy',pred,base)]:
                query=((((x-y)/CAPS)**2).mean(-1)*valid).sum(-1)/valid.sum(-1)
                actual=summarize_query(query,rows)
                if any(not np.isclose(actual[t],v,rtol=2e-5,atol=1e-8) for t,v in report['metrics'][split][name].items()):raise ValueError('Metric mismatch')
            checked+=1
            if split!='inner_held':continue
            planned=np.array([cartesian_residual(base[i,OFFSETS],pred[i,OFFSETS],robots[r['group']]) for i,r in enumerate(rows)])
            vm=labels['direction_valid'][ids];metrics={}
            for teacher in ('raw','bounded'):
                y=labels[teacher+'_direction'][ids]
                metrics[teacher]=geometry_error(planned,y,vm,rows)
                if report['mode']!='bc':
                    head=values['predicted_direction'][:,OFFSETS]
                    query=(((head-y)**2).mean(-1)*vm).sum(-1)/vm.sum(-1)
                    metrics[teacher]['head_mse_all14']=summarize_query(query,rows)
            geometries.append(dict(seed=report['seed'],mode=report['mode'],metrics=metrics))
            print('FK checked',report['seed'],report['mode'],flush=True)
    lookup={(r['seed'],r['mode']):r for r in result['reports']};geo={(r['seed'],r['mode']):r['metrics'] for r in geometries};summary={}
    for mode in ('raw','bounded'):
        summary[mode]={}
        for task in lookup[(0,'bc')]['metrics']['inner_held']['expert_mse']:
            row={}
            for name in ('expert_mse','reachable_mse'):
                old=np.array([lookup[(s,'bc')]['metrics']['inner_held'][name][task] for s in (0,1,2)])
                new=np.array([lookup[(s,mode)]['metrics']['inner_held'][name][task] for s in (0,1,2)])
                row[name]=dict(relative_reduction=(1-new/old).tolist(),mean_relative_reduction=float(np.mean(1-new/old)),better_seeds=int((new<old).sum()))
            row['geometry']={}
            for teacher in ('raw','bounded'):
                row['geometry'][teacher]={}
                for kind in ('translation_m2','rotation_rad2'):
                    old=np.array([geo[(s,'bc')][teacher][kind][task] for s in (0,1,2)])
                    new=np.array([geo[(s,mode)][teacher][kind][task] for s in (0,1,2)])
                    row['geometry'][teacher][kind]=dict(relative_reduction=(1-new/old).tolist(),mean_relative_reduction=float(np.mean(1-new/old)),better_seeds=int((new<old).sum()))
            if task in lookup[(0,'bc')]['metrics']['anchor_held']['residual_energy']:
                old=np.array([lookup[(s,'bc')]['metrics']['anchor_held']['residual_energy'][task] for s in (0,1,2)])
                new=np.array([lookup[(s,mode)]['metrics']['anchor_held']['residual_energy'][task] for s in (0,1,2)])
                row['anchor_deviation_change']=(new/old-1).tolist()
            summary[mode][task]=row
    write_json(a.run/'paired_audit.json',dict(status='passed',verified_prediction_files=checked,summary=summary,geometry=geometries,
        report_code_sha256=digest(Path(__file__)),scope='All valid queries, two arms, five fixed horizons. FK of commands is not measured motion or task success. Same developer scenes across three training seeds; no significance or task generalization claim. Head prediction and actual corrected-action FK are separately measured.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
