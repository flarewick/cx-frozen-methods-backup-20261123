"""Shared observable geometry readout; no actions or task/asset selector inputs."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
from .structured_readout_probe import StructuredProbe
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .prepare_geometry_supervision import CONDITIONS, background_weights
from .diagnose_teacher_observability import balanced
from .train_rotation_curriculum import state_hash


class GeometryReadout(StructuredProbe):
    def __init__(self, channels, mean):
        super().__init__(channels, np.zeros(7), np.zeros(4))
        self.head = nn.Linear(64, 24); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        del self.phase_prior
        self.reg_mean = torch.as_tensor(mean, dtype=torch.float32)

    def forward(self, x):
        cur, cw = self.attention(x, '', 'conditional'); past, pw = self.attention(x, 'past_', 'conditional')
        merged = torch.cat([cur, x['text'].flatten(1), past, x['past_text'].flatten(1),
            x['state'], x['past_state'], x['ee_pose'].flatten(1), x['past_ee_pose'].flatten(1), x['has_history']], 1)
        return dict(regression=self.head(self.trunk(merged)) + self.reg_mean,
                    current_attention=cw, past_attention=pw)


def geometry_loss(pred, target, valid, weights):
    per_row = ((pred - target).square() * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (per_row * weights).sum()


def load(run, learning=True):
    run = Path(run); p = read(run / 'protocol.json'); verify(p)
    with np.load(run / 'inputs.npz') as z: x = {k: z[k].copy() for k in z.files}
    expected = {'visual', 'text', 'state', 'past_visual', 'past_text', 'past_state', 'has_history', 'ee_pose', 'past_ee_pose'}
    if set(x) != expected or set(x) != set(p['forward_keys']): raise ValueError('Unknown model input')
    with np.load(run / 'labels_offline_only.npz') as z: y = {k: z[k].copy() for k in z.files}
    q = None
    if learning:
        q = read(run / 'learning_protocol.json'); verify(q)
    return p, q, x, y


def prepare_learning(run):
    p, _, x, y = load(run, learning=False)
    if not read(run / 'label_audit.json')['passed']: raise ValueError('Audited labels required')
    rows = p['rows']; ei = p['expert_indices']; ri = p['recovery_indices']
    expert_weight = np.zeros(len(rows)); expert_weight[ei] = background_weights([rows[i] for i in ei])
    recovery_weight = np.zeros(len(rows)); recovery_weight[ri] = balanced(rows, ri)
    jobs = []; initializations = []
    for scene in sorted({rows[i]['scene_group'] for i in ri}):
        held = [i for i in ri if rows[i]['scene_group'] == scene]
        remaining = [i for i in ri if i not in held]
        shared_weight = expert_weight + recovery_weight; shared_weight[held] = 0.
        # All conditions share these training-only pooled constants, including the expert mean.
        # Contrast changes expert gradient supervision, not all expert-label information.
        denom = (shared_weight[:, None] * y['regression_valid']).sum(0)
        mean = (shared_weight[:, None] * y['regression_valid'] * y['regression']).sum(0) / np.maximum(denom, 1e-12)
        for condition in CONDITIONS:
            weights = recovery_weight.copy()
            if condition != 'recovery_only_omit': weights += expert_weight
            if condition != 'expert_plus_include': weights[held] = 0.
            jobs.append(dict(scene=scene, condition=condition, weights=weights.tolist(),
                             held=held, held_weights=balanced(rows, held).tolist(),
                             mean=mean.tolist(), train=np.flatnonzero(weights).tolist()))
        for seed in (0, 1, 2):
            torch.manual_seed(seed); model = GeometryReadout(p['channel_dim'], mean)
            file = run / f'initial_{scene.split(":")[-1]}_s{seed}.pt'; torch.save(model.state_dict(), file)
            initializations.append(dict(scene=scene, seed=seed, path=str(file), sha256=digest(file), state_sha256=state_hash(model)))
    code = ['geometry_readout.py', 'train_geometry_readout.py', 'audit_geometry_readout.py',
            'geometry_labels.py', 'prepare_geometry_supervision.py', 'structured_readout_probe.py', 'structured_features.py']
    files = [run / f for f in ('protocol.json', 'label_audit.json', 'inputs.npz', 'labels_offline_only.npz')]
    files += [Path(__file__).with_name(f) for f in code] + [Path(__file__).parents[1] / 'scripts/train_geometry_readout.sh']
    files += [Path(v['path']) for v in initializations]
    write_json(run / 'learning_protocol.json', dict(status='frozen_before_training', jobs=jobs,
        conditions=list(CONDITIONS), seeds=[0, 1, 2], updates=600, lr=.0003, weight_decay=.01,
        parameters=sum(v.numel() for v in model.parameters()), initializations=initializations,
        expert_weights=expert_weight.tolist(), recovery_weights=recovery_weight.tolist(),
        loss='Masked per-row mean MSE, xyz/.1 and root rotation6D; expert task/scene/row weight total1, recovery scene/branch/row total1. Omitted rows zero, no redistribution.',
        common_initial_prior='Same actual initialization and pooled valid-target mean from all32training experts + two other recovery scenes in every condition. Recovery-only control removes expert gradient supervision, not the shared expert-derived prior/PCA.',
        evaluation='Final600, all27 heads. Root xyz, functional/nominal xyz, root orientation/face direction; derive relative geometry analytically. No contact-candidate selection or articulated-progress claims.',
        criterion='At least2of3 held scenes: in all3paired seeds expert_plus_omit primary functional xyz and primary root xyz beat recovery_only_omit and training-task-mean baseline; face angle no more than1degree worse than recovery_only_omit. Signal only, not deployment approval.',
        isolation='expert_plus_include uses held recovery labels solely as fitting upper bound; its held metrics are not generalization. No new action head/adapter/LoRA, simulator, router, TTA or calibration/confirmation scenes.',
        source_sha256={str(f): digest(f) for f in files}))
    print('GEOMETRY_LEARNING_PREPARED', len(jobs) * 3, 'models', sum(v.numel() for v in model.parameters()), flush=True)


def rotations6d(x):
    c = np.asarray(x).reshape(-1, 2, 3, 2)
    a = c[..., 0]; b = c[..., 1]
    na = np.linalg.norm(a, axis=-1); a = a / np.maximum(na[..., None], 1e-12)
    b = b - (b * a).sum(-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1); b = b / np.maximum(nb[..., None], 1e-12)
    r = np.stack([a, b, np.cross(a, b)], -1)
    invalid = (na < 1e-8) | (nb < 1e-8)
    r[invalid] = np.eye(3)
    return r, invalid


def metrics(pred, truth, ids, weights):
    pred = pred[ids]; gt = truth['regression'][ids]; w = np.asarray(weights, float); w = w / w.sum()
    rotations, bad = rotations6d(pred[:, 12:])
    r = truth['root_world'][ids, :, :3, :3]
    angular = np.degrees(Rotation.from_matrix((rotations @ r.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude()).reshape(-1, 2)
    face = np.degrees(np.arccos(np.clip((rotations[..., 0] * r[..., 0]).sum(-1), -1, 1)))
    angular[bad] = 180.; face[bad] = 180.
    root = np.linalg.norm((pred[:, :6] - gt[:, :6]).reshape(-1, 2, 3), axis=-1) * 10
    func = np.linalg.norm((pred[:, 6:12] - gt[:, 6:12]).reshape(-1, 2, 3), axis=-1) * 10
    output = {}
    for slot, name in enumerate(('primary', 'destination')):
        rv = truth['root_valid'][ids, slot]; fv = truth['functional_valid'][ids, slot]
        def average(values, valid):
            return None if not valid.any() else float((w[valid] @ values[valid]) / w[valid].sum())
        output[name] = dict(root_position_cm=average(root[:, slot], rv), functional_position_cm=average(func[:, slot], fv),
                           root_angle_deg=average(angular[:, slot], rv), face_angle_deg=average(face[:, slot], rv),
                           invalid_rotations=int(bad[:, slot][rv].sum()))
    valid = truth['relation_valid'][ids]
    relation = (pred[:, 9:12] - pred[:, 6:9]) * .1
    output['relation_world_error_cm'] = None if not valid.any() else float(w[valid] @ np.linalg.norm(
        relation[valid] - truth['relation_world_m'][ids][valid], axis=1) * 100 / w[valid].sum())
    pp = output['primary']; pp['geometry_gate'] = (pp['root_position_cm'] is not None and pp['functional_position_cm'] is not None
        and pp['root_position_cm'] <= 1 and pp['functional_position_cm'] <= 1 and pp['root_angle_deg'] <= 3 and pp['invalid_rotations'] == 0)
    return output


def task_mean(pred_template, labels, rows, weights, held):
    """A strong diagnostic prior keyed by evaluator task metadata, never model input."""
    output = pred_template.copy()
    for task in {rows[i]['task'] for i in held}:
        ids = [i for i, row in enumerate(rows) if row['task'] == task and weights[i] > 0]
        w = np.asarray(weights)[ids, None] * labels['regression_valid'][ids]
        mean = (w * labels['regression'][ids]).sum(0) / np.maximum(w.sum(0), 1e-12)
        output[[i for i in held if rows[i]['task'] == task]] = mean
    return output
