"""Offline factorized diagnosis: global/object-oracle vision x native/FK actions."""
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .cache_siglip import setup_imports
from .effect_learning_probe import weights,shuffle_within_checkpoint
from .object_alignment import pool_object
from .paths import ROOT,new_run,write_json
from .robot_kinematics import RobotFK,cartesian_residual
from .semantic_effect_probe import squared_distances,rbf,fit_predict
from .short_window_probe import targets,ALPHA


def main():
    run=new_run('representation_probe',group=('effect_learning',));print(run,flush=True)
    fk_audit=json.loads((ROOT/'outputs/reports/audits/fk_camera_check_20260928.json').read_text())
    if fk_audit['status']!='passed':raise ValueError('FK failed camera verification')
    setup_imports(run)
    from openpi.shared.image_tools import resize_with_pad
    cache=ROOT/'outputs/feature_caches/short_siglip_train_v1_gpu'
    cached=json.loads((cache/'result.json').read_text());lookup={(r['job'],r['offset']):r for r in cached['records']}
    horizons=json.loads((ROOT/'outputs/reports/audits/cumulative_probes_20260928.json').read_text())['records']
    first=json.loads((Path(horizons[0]['dataset'])/'index.json').read_text())
    global_features={};object_features={};cartesian={};provenance=[]
    for source in first['sources']:
        path=Path(source['run']);result=json.loads((path/'result.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
        if digest(path/'result.json')!=source['result_sha256']:raise ValueError('Source changed')
        config_path=Path(manifest['source'])/'resolved_config.json';config=json.loads(config_path.read_text());robots=[RobotFK(config,side) for side in ('left','right')]
        provenance.append(dict(job=source['id'],config_sha256=digest(config_path),urdf_sha256=digest(robots[0].urdf)))
        frames={f['offset']:f for f in result['branches']['base']['frames']};mapping=result['branches']['base']['entity_id_map_offline'];vf=[];of={name:[] for name in mapping}
        for offset in (0,-8):
            frame=frames[offset];row=lookup[source['id'],offset]
            if digest(Path(row['file']))!=row['sha256'] or digest(path/frame['offline_file'])!=frame['offline_sha256']:raise ValueError('Feature/mask changed')
            with np.load(row['file']) as z:pooled=z['pooled']
            v=pooled.reshape(3,-1).astype(float);vf.append((v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)).ravel())
            with np.load(path/frame['offline_file']) as z:maps=z['actor_ids']
            for name,ids in mapping.items():
                masks=[]
                for raw in maps:
                    mask=np.repeat((np.isin(raw,ids).astype(np.uint8)*255)[...,None],3,axis=-1)
                    masks.append(np.asarray(resize_with_pad(mask,224,224))[...,0].astype(float)/255)
                of[name].append(pool_object(pooled,np.stack(masks)))
        global_features[source['id']]=np.concatenate(vf)
        for name,pairs in of.items():object_features[source['id'],name]=(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]))
        with np.load(path/'checkpoint.npz') as z:base=z['base_actions']
        for name in result['branches']:
            with np.load(path/f'transition_{name}.npz') as z:proposal=z['proposed_actions']
            cartesian[source['id'],name]=cartesian_residual(base,proposal,robots)
        print('PREPARED '+source['id'],flush=True)
    reports=[]
    for h in horizons:
        dataset=Path(h['dataset']);index=json.loads((dataset/'index.json').read_text());labels=json.loads((dataset/'labels_offline.json').read_text())
        if digest(dataset/'inputs.npz')!=index['inputs_sha256'] or digest(dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Dataset changed')
        with np.load(dataset/'inputs.npz') as z:context=z['context'].astype(float);native=z['action'].astype(float)
        context=context[:,np.r_[0:14,590:618,1194:context.shape[1]]]
        for (task,entity,kind),items in targets(index,labels).items():
            ids=np.array([p[0] for p in items]);y=np.stack([p[1] for p in items]);visible=np.array([p[2] for p in items]);records=[index['records'][i] for i in ids]
            vglobal=np.stack([global_features[r['job']] for r in records]);vobject=np.stack([object_features[r['job'],entity][0] for r in records]);geom=np.stack([object_features[r['job'],entity][1] for r in records])
            dg=squared_distances(vglobal);do=squared_distances(vobject);dp=squared_distances(context[ids]);dggeom=squared_distances(np.c_[context[ids],geom])
            a_native=native[ids];a_cart=np.stack([cartesian[r['job'],r['branch']].copy() for r in records]);a_cart[:,h['horizon']:]=0;a_cart=a_cart.reshape(len(ids),-1)
            shuffled=[shuffle_within_checkpoint(a_cart,records,s) for s in (0,1,2)]
            modes=['zero','no_global','no_object','only_native','only_cartesian','global_native','object_native','global_cartesian','object_cartesian','shuffled_0','shuffled_1','shuffled_2']
            preds={m:np.full_like(y,np.nan) for m in modes}
            for group in sorted({r['group'] for r in records}):
                train=np.array([i for i,r in enumerate(records) if r['group']!=group and visible[i]],int);test=np.array([i for i,r in enumerate(records) if r['group']==group],int)
                if not len(train):continue
                kg=(rbf(dg,train)[0]+rbf(dp,train)[0])/2;ko=(rbf(do,train)[0]+rbf(dggeom,train)[0])/2
                for mode in modes:
                    if mode=='zero':preds[mode][test]=0;continue
                    a=a_native if mode.endswith('native') else shuffled[int(mode[-1])] if mode.startswith('shuffled') else a_cart
                    a=np.c_[a,a*a];ka=a@a.T;ka/=max(float(np.diag(ka)[train].mean()),1e-12)
                    kc=kg if mode.startswith('global') or mode=='no_global' else ko
                    k=kc if mode.startswith('no_') else ka if mode.startswith('only_') else kc*ka
                    preds[mode][test],_=fit_predict(k,y,records,train,test,ALPHA,False)
            valid=np.isfinite(preds['object_cartesian']).all(axis=1);selected=np.flatnonzero(valid)
            if not len(selected):continue
            w=weights([records[i] for i in selected]);mse={m:float(w@np.mean((v[selected]-y[selected])**2,axis=1)) for m,v in preds.items()}
            mse['shuffled_mean']=float(np.mean([mse[f'shuffled_{i}'] for i in (0,1,2)]))
            key=f'h{h["horizon"]}_{len(reports):03d}';np.savez_compressed(run/(key+'.npz'),row_ids=ids,target=y,visible=visible,**preds)
            reports.append(dict(horizon=h['horizon'],task=task,entity=entity,kind=kind,rows=len(ids),visible_rows=int(visible.sum()),mse=mse,improvement={m:1-v/mse['zero'] for m,v in mse.items()} if mse['zero']>1e-16 else None,predictions=key+'.npz'))
        print('COMPLETE horizon '+str(h['horizon']),flush=True)
    write_json(run/'result.json',dict(status='complete',reports=reports,provenance=provenance,alpha=ALPHA,
      scope='Train-scene LOSO oracle entity heads. Object modes use GT masks and are PRIVILEGED diagnostics, not deployable M1. FK modes use robot config/planned actions only. Current and past inputs, no future masks as features.',
      limits='4x4 pooled patches mix foreground/background; oracle masks are not ground-truth pose upper bounds. No shared head or task-held-out claims. Cartesian deltas describe planned link pose, not realized physics or IK corrections.'))
    print('COMPLETE '+str(run),flush=True)

if __name__=='__main__':main()
