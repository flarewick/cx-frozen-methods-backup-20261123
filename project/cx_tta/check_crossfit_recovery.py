"""Post-hoc capacity check from saved S1y weights; never creates router labels."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .time_encoding import TimeEncodedAdapter
from .robot_kinematics import RobotFK
from .candidate_diagnostic import row_metrics, summarize


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();torch.set_num_threads(2);threadpool_limits(2)
    p=read(run/'protocol.json');verify(p);result=read(run/'result.json')
    if not read(run/'candidate_audit.json')['passed']:raise ValueError('Audited candidates required')
    ids=np.array([i for i,r in enumerate(p['rows']) if r['kind']=='recovery'])
    rows=[p['rows'][i] for i in ids]
    with np.load(run/'data.npz') as z:base,target,valid=(z[k][ids].copy() for k in ('base','target','valid'))
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    baseline=row_metrics(base.astype(float),target,valid,robots);reports=[]
    for source in result['reports']:
        if digest(Path(source['checkpoint']))!=source['checkpoint_sha256']:raise ValueError('Changed candidate checkpoint')
        job=next(j for j in p['jobs'] if j['name']==source['job'])
        with np.load(job['context']) as z:context=z['context'][ids].copy()
        model=TimeEncodedAdapter(157,encoding=source['encoding'],envelope=True)
        model.load_state_dict(torch.load(source['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():prediction=model(torch.from_numpy(context),torch.from_numpy(base)).numpy()
        metric=row_metrics(prediction.astype(float),target,valid,robots)
        train=[j for j,i in enumerate(ids) if int(i) in job['train']]
        excluded=[j for j,i in enumerate(ids) if int(i) not in job['train']]
        for name,subset in [('train_recovery',train),('excluded_recovery',excluded)]:
            if not subset:continue
            summary=summarize(metric,prediction,target,valid,rows,subset,baseline)
            reports.append(dict(job=source['job'],encoding=source['encoding'],seed=source['seed'],part=name,
                groups=[g['group'] for g in summary['groups']],summary=summary))
    verify(p)
    write_json(run/'recovery_fit_check.json',dict(status='complete',reports=reports,
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'candidate_audit.json',Path(__file__)]},
        scope='Post-hoc diagnostic only from all30saved weights, no updates. In-fit recovery results MUST NOT train '
              'the router. In-fit versus excluded scenes are different scenes, and comparison to old S1v changes '
              'expert/PCA/anchor support too. This is not a controlled causal attribution to data quantity or representation.'))
    for part in ('train_recovery','excluded_recovery'):
        for encoding in p['encodings']:
            found=[r for r in reports if r['part']==part and r['encoding']==encoding]
            print(part,encoding,sum(r['summary']['recovery_fit_gate'] for r in found),'/',len(found),flush=True)


if __name__=='__main__':main()
