"""Validate wide-data loss masking, component weights and control isolation."""
import numpy as np
import torch
from .check_time_encoding import numpy_temporal
from .recovery_temporal import temporal_loss
from .time_encoding import TimeEncodedAdapter
from .train_recovery_fit import end_torch


def check_interface(p, data, weights_np, device, per_row_loss, robots):
    records=[]
    for condition,setting in p['conditions'].items():
        ix=np.flatnonzero(weights_np[condition]);local={int(g):i for i,g in enumerate(ix)}
        allowed=sorted(sum([p['parts'][c] for c in setting['components']],[]))
        if ix.tolist()!=allowed or len(allowed)!=len(set(allowed)):
            raise ValueError('Training rows duplicated or changed')
        expected=np.zeros(len(p['rows']),np.float32)
        for part in setting['components']:
            ids=p['parts'][part];tasks={p['rows'][i]['task'] for i in ids}
            values=[]
            for i in ids:
                row=p['rows'][i]
                groups={p['rows'][j]['group'] for j in ids if p['rows'][j]['task']==row['task']}
                count=sum(p['rows'][j]['group']==row['group'] for j in ids)
                values.append(1/(len(tasks)*len(groups)*count))
            values=np.array(values);expected[ids]=values/values.sum()
        if not np.array_equal(expected,weights_np[condition]):raise ValueError('Component weighting mismatch')
        selected_pairs=p['seam_pairs'] if 'recovery' in setting['components'] else []
        pairs_np=np.array([[local[x['before']],x['last'],local[x['after']]] for x in selected_pairs],dtype=np.int64).reshape(-1,3)
        pairs=torch.as_tensor(pairs_np,device=device)
        pw=torch.tensor(p['seam_weights'] if selected_pairs else [],device=device,dtype=torch.float32)
        idx=torch.as_tensor(ix,device=device);weights=torch.as_tensor(expected[ix],device=device)
        valid=data['valid'][idx];target_all=data['target']
        torch.manual_seed(710)
        model=TimeEncodedAdapter(p['context_dim'],encoding=setting['encoding'],envelope=True).to(device)
        with torch.no_grad():
            if not torch.equal(model(data['semantic'][idx],data['base'][idx]),data['base'][idx]):
                raise ValueError('Wide zero-head identity failed')
            model.head.weight.normal_(0,.06);model.head.bias.normal_(0,.06)
        def loss_grad(labels):
            model.zero_grad();target=labels[idx]
            pred=model(data['semantic'][idx],data['base'][idx])
            dense=(per_row_loss(pred,target,valid,end_torch(target,robots),model,'se3')[0]*weights).sum()
            within,seam=temporal_loss(pred,target,valid,model.caps,weights,pairs,pw)
            loss=dense+within+seam;loss.backward()
            gradient=torch.cat([v.grad.flatten() for v in model.parameters()]).detach().clone()
            return loss.detach(),gradient,pred.detach(),(within.detach(),seam.detach())
        loss,gradient,pred,terms=loss_grad(target_all)
        changed=target_all.clone();changed[~data['valid']]=1000
        masked_loss,masked_gradient,*_=loss_grad(changed)
        if not torch.equal(loss,masked_loss) or not torch.equal(gradient,masked_gradient):
            raise ValueError('Invalid target tail affects training')
        changed=target_all.clone();inactive=torch.ones(len(target_all),device=device,dtype=torch.bool);inactive[idx]=False
        changed[inactive]=1000
        isolated_loss,isolated_gradient,*_=loss_grad(changed)
        if not torch.equal(loss,isolated_loss) or not torch.equal(gradient,isolated_gradient):
            raise ValueError('Excluded recovery/held labels affect training')
        if not torch.isfinite(gradient).all() or gradient.norm()==0:raise ValueError('Invalid wide training gradient')
        extra_grad=float(model.trunk[0].weight.grad[:,-8:].norm())
        if extra_grad<=0 or not np.isfinite(extra_grad):raise ValueError('Time columns lack gradient')
        values=[x.detach().cpu().numpy() for x in (pred,target_all[idx],valid,model.caps,weights,pairs,pw)]
        reference=numpy_temporal(*values);actual=np.array([float(x) for x in terms])
        if not np.allclose(reference,actual,rtol=2e-6,atol=2e-6):raise ValueError('Wide temporal aggregation mismatch')
        if not selected_pairs and float(terms[1])!=0:raise ValueError('Control received recovery seam supervision')
        records.append(dict(condition=condition,training_rows=len(ix),valid_actions=int(valid.sum()),
            seam_pairs=len(selected_pairs),loss=float(loss),finite_nonzero_gradient=True,time_column_gradient_norm=extra_grad,
            invalid_tail_loss_gradient_exact=True,excluded_labels_loss_gradient_exact=True,zero_head_exact=True,
            component_weight_sums={c:float(expected[p['parts'][c]].astype(float).sum()) for c in setting['components']},
            numpy_temporal_max_error=float(np.abs(reference-actual).max())))
    return dict(passed=True,checks=records,scope='Actual wide training data and optimizer-loss isolation checks; no task performance claim.')
