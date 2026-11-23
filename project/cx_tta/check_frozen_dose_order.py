"""Post-hoc development control: vary dose ordering with frozen fold thresholds."""
import argparse
from pathlib import Path
import numpy as np
from .paths import write_json
from .audit import digest
from .probe_candidates import read
from .diagnose_component_calibration import metrics


def main(run):
    p=read(run/'protocol.json');source=Path(p['source']);rows=p['rows'];ids=p['source_indices']
    reports=read(run/'result.json')['reports'];sources=read(source/'probe_result.json')['reports']
    # This control was specified after inspecting S1z; it is not an independent confirmation.
    write_json(run/'dose_control_protocol.json',dict(status='frozen_before_dose_control_results',
        design='Keep all existing per-fold thresholds; replace ranking only with smallest allowed positive alpha, '
               'or alpha=.25 if that particular dose passes its own fixed score/risk tests, otherwise off. '
               'No threshold recalibration, no retraining, no S1y validation.',
        scope='Post-hoc development attribution check; not a selected deployable rule or new confirmation.',
        source_sha256={str(f):digest(f) for f in (Path(__file__),run/'protocol.json',run/'result.json')}))
    labels={}
    for entry in read(source/'probe_data_manifest.json')['records']:
        with np.load(entry['file']) as z:labels[entry['encoding'],entry['support']]={k:z[k][:,ids].copy() for k in z.files}
    out=[]
    for r,s in zip(reports,sources):
        with np.load(s['artifact']) as z:prediction=z['prediction'][:,ids].copy()
        panel=labels[s['encoding'],s['support']]
        result={k:s[k] for k in ('encoding','support','architecture','seed')};result['rules']={}
        for rule,report in r['rules'].items():
            head=0 if rule.startswith('gain_') else 1
            low=np.zeros(prediction.shape[:-2],np.int16);quarter=low.copy()
            for fold in report['folds']:
                held=[i for i,row in enumerate(rows) if row['group']==fold['held_scene']]
                for c,setting in enumerate(fold['thresholds']):
                    t,risk=setting['threshold'],setting['risk']
                    if t is None:continue
                    pred=prediction[:,held,:,c]
                    allowed=(pred[...,head]>t)&(pred[...,2]<=risk)
                    choice=allowed.argmax(-1)+1;choice[~allowed.any(-1)]=0
                    low[:,held,:,c]=choice
                    quarter[:,held,:,c]=allowed[...,0].astype(np.int16)
            if np.any((low>0)!=(np.load(run/f"{s['encoding']}_{s['support']}_{s['architecture']}_s{s['seed']}.npz")[rule]>0)):
                raise ValueError('Smallest allowed dose must preserve activation coverage')
            result['rules'][rule]=dict(smallest_allowed=metrics(panel,low,rows,range(len(rows))),
                quarter_only=metrics(panel,quarter,rows,range(len(rows))))
        out.append(result)
    cp=read(run/'dose_control_protocol.json')
    if any(digest(Path(f))!=sha for f,sha in cp['source_sha256'].items()):raise ValueError('Changed source')
    write_json(run/'dose_controls.json',dict(status='complete',reports=out,scope=cp['scope'],
        activation_coverage_checks=len(out)*len(out[0]['rules']),passed=True))
    print('DOSE_CONTROLS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
