"""Compare fixed-input numerical probes without interpreting them as policy scores."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .paths import write_json, ROOT


def difference(x,y):
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite, matching action shapes required')
    delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {'equal':bool(np.array_equal(x,y)),'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'different_entries':int(np.count_nonzero(x!=y))}


def compare(a,b,ra=0,rb=0):
    ma,mb=[json.loads((p/'manifest.json').read_text()) for p in (a,b)]
    if ma['input_manifest_sha256']!=mb['input_manifest_sha256']:
        raise ValueError('Input bundles must match')
    aa,bb=[{(r['repeat'],r['input']):r for r in json.loads((p/'result.json').read_text())['records']} for p in (a,b)]
    if {i for r,i in aa if r==ra}!={i for r,i in bb if r==rb}:raise ValueError('Incomplete inputs')
    rows=[]
    for i in sorted(i for r,i in aa if r==ra):
        x,y=aa[ra,i],bb[rb,i]
        rows.append({'input':i,'transformed_equal':x['transformed']==y['transformed'],
                     'rng_before_equal':x['rng_before']==y['rng_before'],
                     **difference(np.load(a/f'r{ra}_input{i}.npy'),np.load(b/f'r{rb}_input{i}.npy'))})
    return {'a':str(a),'b':str(b),'repeats':[ra,rb],'rows':rows,
            'all_inputs_equal':all(r['transformed_equal'] and r['rng_before_equal'] for r in rows),
            'equal_outputs':sum(r['equal'] for r in rows),'total_inputs':len(rows),
            'max_abs':max(r['max_abs'] for r in rows)}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();runs=[r.resolve() for r in args.runs]
    result={'scope':'fixed-input numerical diagnostic; not task success evaluation',
            'within_process':[compare(r,r,0,1) for r in runs],
            'cross_process':[compare(a,b) for a,b in itertools.combinations(runs,2)],
            'manifests':{str(r):json.loads((r/'manifest.json').read_text()) for r in runs},
            'rng_consumed':{str(r):sum(v['rng_before']!=v['rng_after'] for v in json.loads((r/'result.json').read_text())['records']) for r in runs}}
    write_json(args.out.resolve(),result)
    for kind in ('within_process','cross_process'):
        for r in result[kind]:print(kind,Path(r['a']).name,Path(r['b']).name,'inputs_equal',r['all_inputs_equal'],'outputs_equal',r['equal_outputs'],'/',r['total_inputs'],'max_abs',r['max_abs'])
if __name__=='__main__':main()
