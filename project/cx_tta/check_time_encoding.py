"""Actual-data loss, padding, gradient and zero-adapter interface checks."""
import numpy as np
import torch
from .recovery_temporal import temporal_loss
from .time_encoding import TimeEncodedAdapter, time_features
from .probe_recovery_time import numpy_time
from .train_recovery_fit import end_torch


def numpy_temporal(pred, target, valid, caps, weights, pairs, pair_weights):
    within = 0.
    for i in range(len(pred)):
        errors = []
        for j in range(1, pred.shape[1]):
            if valid[i, j-1] and valid[i, j]:
                difference = (pred[i, j]-pred[i, j-1]) - (target[i, j]-target[i, j-1])
                errors.append(float(np.mean((difference/caps)**2)))
        if errors:
            within += float(weights[i]) * float(np.mean(errors))
    seam = 0.
    for (before, last, after), weight in zip(pairs, pair_weights):
        difference = (pred[after, 0]-pred[before, last]) - (target[after, 0]-target[before, last])
        seam += float(weight) * float(np.mean((difference/caps)**2))
    return np.array([within, seam])


def check_interface(p, data, device, per_row_loss, robots):
    ids = sorted(sum([p['parts'][k] for k in ('recovery', 'ordinary', 'anchor_fit')], []))
    local = {g:i for i,g in enumerate(ids)}
    pairs_np = np.array([[local[x['before']], x['last'], local[x['after']]] for x in p['seam_pairs']])
    pairs = torch.as_tensor(pairs_np, device=device)
    pw = torch.tensor(p['seam_weights'], device=device, dtype=torch.float32)
    weights_np = np.zeros(len(ids), np.float32)
    for part in ('recovery', 'ordinary', 'anchor_fit'):
        part_ids = p['parts'][part]
        tasks = {p['rows'][i]['task'] for i in part_ids}
        for i in part_ids:
            row = p['rows'][i]
            ng = len({p['rows'][j]['group'] for j in part_ids if p['rows'][j]['task']==row['task']})
            nr = sum(p['rows'][j]['group']==row['group'] for j in part_ids)
            weights_np[local[i]] = 1/(len(tasks)*ng*nr)
    weights = torch.as_tensor(weights_np, device=device)
    target, valid = data['target'][ids], data['valid'][ids]
    checks = []
    for encoding, envelope in ((m,e) for m in ('repeat','fourier') for e in (False,True)):
        basis = time_features(32, encoding, device=device, dtype=torch.float32).cpu().numpy()
        basis_error = float(np.abs(basis - numpy_time(32, encoding)).max())
        if basis_error > 3e-6:
            raise ValueError('Relative time basis disagrees with independent NumPy formula')
        torch.manual_seed(710)
        model = TimeEncodedAdapter(p['context_dim'], encoding=encoding, envelope=envelope).to(device)
        with torch.no_grad():
            for start in range(0, len(data['base']), 64):
                context, base = data['semantic'][start:start+64], data['base'][start:start+64]
                if not torch.equal(model(context, base), base):
                    raise ValueError('Zero-head identity failed')
                if not torch.equal(model(context, base, enabled=False), base):
                    raise ValueError('Disabled identity failed')
            model.head.weight.normal_(0, .06); model.head.bias.normal_(0, .06)
        def loss_grad(t):
            model.zero_grad()
            out = model(data['semantic'][ids], data['base'][ids])
            pose = (per_row_loss(out, t, valid, end_torch(t, robots), model, 'se3')[0]*weights).sum()
            within, seam = temporal_loss(out, t, valid, model.caps, weights, pairs, pw)
            loss = pose+within+seam; loss.backward()
            grads = torch.cat([v.grad.flatten() for v in model.parameters()]).detach().clone()
            grip_grad = model.head.weight.grad[[6,13]].norm(dim=1).detach().cpu().tolist()
            return loss.detach(), grads, out.detach(), (within.detach(), seam.detach()), grip_grad
        loss, gradient, out, terms, grip_grad = loss_grad(target)
        changed = target.clone(); changed[~valid] = 1000
        padded_loss, padded_gradient, *_ = loss_grad(changed)
        if not torch.equal(loss, padded_loss) or not torch.equal(gradient, padded_gradient):
            raise ValueError('Padding leaked into dense/temporal loss or gradient')
        if not torch.isfinite(gradient).all() or gradient.norm()==0 or min(grip_grad)<=0:
            raise ValueError('Invalid learning gradient')
        extra_gradient = float(model.trunk[0].weight.grad[:, -8:].norm())
        if not np.isfinite(extra_gradient) or extra_gradient <= 0:
            raise ValueError('New time channels have no finite learning gradient')
        exact = temporal_loss(target, target, valid, model.caps, weights, pairs, pw)
        if any(float(x)!=0 for x in exact):
            raise ValueError('Real nonzero teacher increment wrongly penalized')
        values = [x.detach().cpu().numpy() for x in (out,target,valid,model.caps,weights,pairs,pw)]
        expected = numpy_temporal(*values)
        actual = np.array([float(x) for x in terms])
        if not np.allclose(actual, expected, rtol=2e-6, atol=2e-6):
            raise ValueError('Independent temporal aggregation mismatch')
        if envelope:
            base = data['base'][ids][..., [6,13]]
            grip = out[..., [6,13]]
            if ((grip < torch.minimum(base, torch.zeros_like(base))) |
                (grip > torch.maximum(base, torch.ones_like(base)))).any():
                raise ValueError('Additional gripper overshoot')
            plain = TimeEncodedAdapter(p['context_dim'], encoding=encoding, envelope=False).to(device)
            plain.load_state_dict(model.state_dict())
            arm = [0,1,2,3,4,5,7,8,9,10,11,12]
            with torch.no_grad():
                if not torch.equal(plain(data['semantic'][ids],data['base'][ids])[...,arm],out[...,arm]):
                    raise ValueError('Gripper decoder changed arm forward')
        checks.append(dict(encoding=encoding, envelope=envelope, loss=float(loss), finite_nonzero_gradient=True,
            time_basis_numpy_max_error=basis_error, time_channel_gradient_norm=extra_gradient,
            gripper_head_gradient_norms=grip_grad, invalid_tail_loss_and_gradient_exact=True,
            true_teacher_temporal_loss_zero=True, zero_and_disabled_all_rows_exact=True,
            numpy_temporal_max_error=float(np.abs(actual-expected).max())))
    return dict(passed=True, actual_fit_rows=len(ids), pairs=len(pairs_np), checks=checks,
        scope='Interface checks on actual data, not evidence of task performance.')
