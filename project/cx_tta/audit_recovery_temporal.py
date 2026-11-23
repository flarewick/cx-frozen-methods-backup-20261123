"""Extend independent S1s FK audit with decoder, seam, and baseline parity."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json
from .audit_recovery_fit import main as audit_dense
from .check_recovery_temporal import numpy_temporal


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    p, result = read(run/'protocol.json'), read(run/'result.json')
    if not read(run/'loss_interface_check.json')['passed']:
        raise ValueError('Real-data gradient checks failed')
    # Reuse unchanged independent dense FK implementation, not trainer metrics.
    original = sys.argv; sys.argv = [original[0], str(run)]
    try:
        audit_dense()
    finally:
        sys.argv = original
    with np.load(run/'data.npz') as z:
        data = {k:z[k].copy() for k in z.files}
    preflight = read(Path(p['command_audit'])/'result.json')
    if p['seam_pairs'] != preflight['pairs']:
        raise ValueError('Changed audited seam labels')
    source = Path(p['baseline_source']); records = []; replay_count = 0
    for report in result['reports']:
        setting = p['conditions'][report['condition']]
        for entry in report['files']:
            with np.load(entry['path']) as z:
                pred, ids, valid, target = z['predicted'].copy(), z['indices'].copy(), z['valid'].copy(), z['target'].copy()
            if setting['envelope']:
                base = data['base'][ids][..., [6,13]]; grip = pred[..., [6,13]]
                if ((grip<np.minimum(base,0))|(grip>np.maximum(base,1))).any():
                    raise ValueError('Saved gripper output exceeds base-preserving envelope')
            if report['condition']=='baseline':
                f = source/f"semantic_mixed_se3_s{report['seed']}_u{report['update']}_{entry['part']}.npz"
                with np.load(f) as z:
                    if not np.array_equal(pred,z['predicted']):
                        raise ValueError('Baseline replay differs from original S1s predictions')
                replay_count += 1
            if entry['part']=='recovery':
                local = {int(g):i for i,g in enumerate(ids)}
                pairs = np.array([[local[x['before']],x['last'],local[x['after']]] for x in p['seam_pairs']])
                for before,last,after in pairs:
                    a,b = p['rows'][ids[before]],p['rows'][ids[after]]
                    if a['group']!=b['group'] or a['frame']+last+1!=b['frame'] or not(valid[before,last] and valid[after,0]):
                        raise ValueError('Invalid boundary pairing')
                # Equal scene weight, split over blocks; these are diagnostics,
                # not the complete three-component training loss.
                w = np.array([1/(2*sum(p['rows'][j]['group']==p['rows'][i]['group'] for j in ids)) for i in ids])
                loss = numpy_temporal(pred,target,valid,np.array([.1]*6+[.2]+[.1]*6+[.2]),w,pairs,np.array(p['seam_weights']))
                records.append(dict(condition=report['condition'],seed=report['seed'],update=report['update'],
                    recovery_within_difference_loss=float(loss[0]), recovery_seam_difference_loss=float(loss[1])))
    if replay_count!=33:
        raise ValueError('Baseline snapshot/part coverage incomplete')
    write_json(run/'temporal_audit.json',dict(passed=True, dense_audit_passed=True,
        baseline_saved_predictions_exact=replay_count, envelope_checked_every_saved_part=True,
        temporal_records=records, source_sha256={str(f):digest(f) for f in
            [run/'audit.json',run/'loss_interface_check.json',run/'protocol.json',Path(__file__)]},
        scope='Offline fit and command-interface evidence only; no deployment or success claim.'))
    print(json.dumps(dict(temporal_audit_passed=True,baseline_exact_files=replay_count,temporal_records=len(records))))


if __name__=='__main__':
    main()
