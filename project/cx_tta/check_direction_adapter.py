"""Real-data CPU integration only: four tasks, safe inputs, gradient and reload checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .dataset import load_cache
from .direction_adapter import DirectionAdapter, losses
from .paths import new_run, write_json


def load_training_batch(index, samples_per_task=8):
    parts = []
    for task in sorted({r['task'] for r in index['records']}):
        row = next(r for r in index['records'] if r['task']==task and r['status']=='prepared')
        if row['split']!='train' or digest(Path(row['cache']))!=row['cache_sha256'] or digest(Path(row['label']))!=row['label_sha256']:
            raise ValueError('Changed or held-out sample')
        cache = load_cache(Path(row['cache']))
        with np.load(row['label']) as z:
            labels = dict(z)
        if not np.array_equal(labels['source_frame_index'],cache['frame_index']):
            raise ValueError('Label/input alignment error')
        n = min(samples_per_task,len(cache['context']))
        parts.append((row, [cache['context'][:n], cache['base_action'][:n], cache['target'][:n], cache['valid'][:n], labels['expert_plan_direction'][:n], labels['direction_valid'][:n]]))
    tensors = [torch.from_numpy(np.concatenate([x[1][k] for x in parts])) for k in range(6)]
    return tensors, [dict(task=row['task'],seed=row['seed'],samples=len(data[0])) for row,data in parts]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    index=json.loads((a.dataset/'index.json').read_text())
    if index['status']!='complete':raise ValueError('Incomplete dataset')
    batch,provenance=load_training_batch(index)
    output=new_run('direction_adapter_check',group=('reports','adapter_checks'))
    reports=[]
    for mode in ('bc','auxiliary','conditioned'):
        torch.manual_seed(0);model=DirectionAdapter(mode=mode)
        with torch.no_grad():
            prediction,delta,direction=model(*batch[:2])
            if not torch.equal(prediction,batch[1]) or delta.count_nonzero() or direction.count_nonzero():raise ValueError('Zero initialization violated')
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        for step in range(3):
            opt.zero_grad();loss,detail=losses(model,*batch);loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(model.head.weight.grad).all():raise ValueError('Nonfinite optimization')
            action_grad=float(model.head.weight.grad.norm());direction_grad=0. if model.direction_head.weight.grad is None else float(model.direction_head.weight.grad.norm())
            opt.step()
        if action_grad<=0 or (mode!='bc' and direction_grad<=0):raise ValueError('Missing real-data gradient')
        model.eval()
        with torch.no_grad():
            pred,delta,_=model(*batch[:2]);bypass=model(*batch[:2],enabled=False)[0]
            if not torch.equal(bypass,batch[1]) or not torch.isfinite(pred).all() or torch.any(delta.abs()>model.caps+1e-7):raise ValueError('Bound/bypass violated')
        torch.save(model.state_dict(),output/f'{mode}_integration_only.pt')
        restored=DirectionAdapter(mode=mode);restored.load_state_dict(torch.load(output/f'{mode}_integration_only.pt',weights_only=True));restored.eval()
        with torch.no_grad():
            if not torch.equal(restored(*batch[:2])[0],pred):raise ValueError('Reload mismatch')
        reports.append(dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),last_training_loss=float(loss.detach()),action_gradient_norm=action_grad,direction_gradient_norm=direction_grad,steps=3))
    assert len({r['parameters'] for r in reports})==1
    write_json(output/'result.json',dict(status='passed',dataset=str(a.dataset),dataset_index_sha256=digest(a.dataset/'index.json'),samples=provenance,reports=reports,model_code_sha256=digest(Path(__file__).with_name('direction_adapter.py')),scope='Integration only on actual expert observations: exact zero init/bypass, bounded outputs, finite gradients, matched declared parameter counts, exact checkpoint reload. Coarse B2 context; language only implicit in frozen instruction-conditioned base action. Teacher labels enter loss only. No learned semantic object goal, no failure recovery, no deployment or efficacy claim. In BC and auxiliary controls, directional conditioning weights are dormant; report effective capacity in later comparisons.'))
    print(output,flush=True)

if __name__=='__main__':main()
