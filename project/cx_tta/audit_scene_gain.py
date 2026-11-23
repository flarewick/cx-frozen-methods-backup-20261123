"""Independent native labels, probe sampling/reload and calibration isolation checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .audit import digest
from .paths import write_json
from .probe_candidates import read, verify
from .probe_scene_gain import load_inputs
from .scene_gain_probe import observable_features, GainProbe, row_weights, calibrate, select, chosen_actions
from .candidate_diagnostic import CAPS, ALPHAS, row_metrics
from .observable_candidate_scores import weighted_auc
from .recovery_fit_metrics import endpoints
from .robot_kinematics import RobotFK


def label_audit(data, proposals, support, panel, robots):
    base, target, valid = data['base'].astype(float), data['target'].astype(float), data['valid']
    tp = endpoints(target, robots)
    bm = row_metrics(base, target, valid, robots, tp)
    count = valid.sum(1)[:,None]
    mean = lambda v: (v * valid[:,:,None]).sum(1) / count
    def losses(action, m):
        difference = (action-target).reshape(len(base),32,2,7)
        arm = mean((difference[...,:6]**2).mean(-1)) * (6/7) / .1**2 + m['rotation_loss'] + m['position_loss']
        grip = mean(difference[...,6]**2)
        return arm, grip
    ba, bg = losses(base, bm)
    maximum = 0.
    for seed in range(3):
        delta = proposals[seed].astype(float)-base
        if support=='capped':delta=np.clip(delta,-CAPS,CAPS)
        for ai, alpha in enumerate(ALPHAS):
            action=base+alpha*delta
            m=row_metrics(action,target,valid,robots,tp)
            arm,grip=losses(action,m)
            gain=np.stack(((ba-arm)/(1+ba),((bg-grip)/.2**2)/(1+bg/.2**2)),axis=-1)
            positive_arm=((ba-arm)>np.maximum(1e-8,.05*ba)) & (m['angle_deg']<=bm['angle_deg']+1e-6) & (m['position_m']<=bm['position_m']+1e-9) & (m['joint_limit_violations']<=bm['joint_limit_violations'])
            positive_grip=((bg-grip)>np.maximum(1e-8,.05*bg)) & (m['gripper_mae']<=bm['gripper_mae']+1e-8) & (m['gripper_outside_margin']<=bm['gripper_outside_margin'])
            bad_arm=(m['angle_deg']>np.maximum(bm['angle_deg'],1.)+1e-6) | (m['position_m']>np.maximum(bm['position_m'],.01)+1e-9) | (m['joint_limit_violations']>bm['joint_limit_violations'])
            bad_grip=(m['gripper_mae']>np.maximum(bm['gripper_mae'],.02)+1e-8) | (m['gripper_outside_margin']>bm['gripper_outside_margin'])
            error=float(np.abs(gain-panel['gain'][seed,...,ai]).max())
            maximum=max(maximum,error)
            if error>1e-8:raise ValueError('Independent normalized gain mismatch')
            np.testing.assert_array_equal(np.stack((positive_arm,positive_grip),-1),panel['benefit'][seed,...,ai])
            np.testing.assert_array_equal(np.stack((bad_arm,bad_grip),-1),panel['harm'][seed,...,ai])
    return maximum


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run.resolve();torch.set_num_threads(2);threadpool_limits(2)
    p,data,context,proposals=load_inputs(run);cp=read(run/'probe_protocol.json');verify(cp)
    result=read(run/'probe_result.json');robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    maxima=dict(label_gain=0.,cpu_prediction=0.,auc=0.,scalar_fk=0.)
    reports=0;isolations=0
    for entry in read(run/'probe_data_manifest.json')['records']:
        if digest(Path(entry['file']))!=entry['sha256']:raise ValueError('Changed labels')
        encoding,support=entry['encoding'],entry['support']
        with np.load(entry['file']) as z:panel={k:z[k].copy() for k in z.files}
        maxima['label_gain']=max(maxima['label_gain'],label_audit(data,proposals[encoding],support,panel,robots))
        features,_=observable_features(context,data['base'],proposals[encoding],support)
        flat=features.reshape(-1,234)
        mass=np.broadcast_to(row_weights(p['rows'],p['parts']['gate_fit'])[None,:,None,None,None],features.shape[:-1]).ravel().copy()
        mass/=mass.sum()
        mean=np.average(flat,axis=0,weights=mass).astype(np.float32)
        std=np.sqrt(np.average((flat-mean)**2,axis=0,weights=mass)).clip(.05).astype(np.float32)
        for report in [r for r in result['reports'] if r['encoding']==encoding and r['support']==support]:
            if digest(Path(report['artifact']))!=report['sha256'] or digest(Path(report['checkpoint']))!=report['checkpoint_sha256']:
                raise ValueError('Changed probe checkpoint or predictions')
            with np.load(report['artifact']) as z:artifact={k:z[k].copy() for k in z.files}
            for key,value in [('mean',mean),('std',std)]:np.testing.assert_array_equal(artifact[key],value)
            counts=artifact['sampled_counts']
            if np.any(counts[mass==0]) or counts.sum()!=600*512:raise ValueError('Calibration/validation rows sampled for training')
            model=GainProbe(234,report['architecture'])
            model.load_state_dict(torch.load(report['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():
                values=[]
                for start in range(0,len(flat),512):
                    raw=model(torch.from_numpy(((flat[start:start+512]-mean)/std).astype(np.float32)))
                    raw[:,1:]=raw[:,1:].sigmoid();values.append(raw.numpy())
            prediction=np.concatenate(values).reshape(*features.shape[:-1],3)
            error=float(np.abs(prediction-artifact['prediction']).max())
            maxima['cpu_prediction']=max(maxima['cpu_prediction'],error)
            if error>5e-5:raise ValueError('CPU probe reload mismatch')
            # Recalibration must be unchanged when every validation label is corrupted.
            poison={k:v.copy() for k,v in panel.items()};ix=p['parts']['validation']
            poison['gain'][:,ix]+=13.
            poison['benefit'][:,ix]=~poison['benefit'][:,ix]
            poison['harm'][:,ix]=~poison['harm'][:,ix]
            best,_=calibrate(artifact['prediction'],poison,p['rows'],p['parts']['calibration'])
            if best!=report['calibration']:raise ValueError('Validation labels influenced calibration')
            isolations+=1
            choices=np.zeros(artifact['choices'].shape,np.int16) if best['gain_threshold'] is None else select(artifact['prediction'],best['gain_threshold'],best['risk_threshold'])
            np.testing.assert_array_equal(choices,artifact['choices'])
            w=row_weights(p['rows'],p['parts']['validation'],'expert')
            for component,name in enumerate(('arm','grip')):
                weights=np.broadcast_to(w[None,:,None,None],(3,len(w),2,3)).ravel()
                label=panel['benefit'][:,:,:,component,1:].ravel()
                score=artifact['prediction'][:,:,:,component,:,1].ravel()
                yes=(weights>0)&label;no=(weights>0)&~label
                if yes.any() and no.any():
                    mass2=weights[yes,None]*weights[None,no];diff=score[yes,None]-score[None,no]
                    auc=float((mass2*((diff>0)+.5*(diff==0))).sum()/mass2.sum())
                    error=abs(auc-report['validation_rank'][name]['benefit_auc']);maxima['auc']=max(maxima['auc'],error)
                    if error>1e-12:raise ValueError('Independent validation AUC mismatch')
            # Real chosen commands, first/last valid action of one row per validation scene.
            groups=sorted({p['rows'][i]['group'] for i in ix})
            sample=np.array([next(i for i in ix if p['rows'][i]['group']==g) for g in groups])
            action=chosen_actions(data['base'][sample],proposals[encoding][0,sample],choices[0,sample],support)
            vector=endpoints(action,robots)
            for j,source in enumerate(sample):
                for t in (0,int(data['valid'][source].sum())-1):
                    for side,robot in enumerate(robots):
                        difference=float(np.abs(vector[j,t,side]-robot.pose(action[j,t,7*side:7*side+6])).max())
                        maxima['scalar_fk']=max(maxima['scalar_fk'],difference)
                        if difference>1e-10:raise ValueError('Independent chosen-action FK mismatch')
            reports+=1
        print('PROBE_AUDIT',encoding,support,flush=True)
    if reports!=24:raise ValueError('Incomplete24probe matrix')
    verify(p);verify(cp)
    write_json(run/'probe_audit.json',dict(passed=True,models=reports,validation_poison_calibration_checks=isolations,
        independent_label_conditions=4,maxima=maxima,source_sha256={str(f):digest(f) for f in
            [run/'protocol.json',run/'probe_protocol.json',run/'candidate_audit.json',run/'probe_data_manifest.json',run/'probe_result.json',Path(__file__)]},
        scope='Source/label/selection/calibration/numerical checks, not a deployment or physical success claim.'))
    print('PROBE_AUDIT_PASS',maxima,flush=True)


if __name__=='__main__':main()
