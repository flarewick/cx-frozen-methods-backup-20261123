"""S2h budget ablation: restart identical initial models, change only600->1400updates."""
import argparse
import copy
from pathlib import Path
from .paths import ROOT, new_run, write_json
from .audit import digest
from .probe_candidates import read, verify


def prepare(source):
    source = source.resolve()
    p = read(source / 'protocol.json'); q = read(source / 'learning_protocol.json')
    verify(p); verify(q)
    for f in ('label_audit.json', 'learning_preflight.json', 'learning_audit.json'):
        assert read(source / f)['passed']; verify(read(source / f))
    result = read(source / 'learning_result.json')
    assert result['status'] == 'complete' and len(result['reports']) == 27 and q['updates'] == 600
    run = new_run('geometry_budget', group=('feature_caches',)); print('RUN', run, flush=True)
    for name in ('inputs.npz', 'labels_offline_only.npz'):
        (run / name).symlink_to(source / name)
    pp = copy.deepcopy(p)
    pp['budget_control_parent'] = str(source)
    pp['source_sha256'].update({str(source / 'protocol.json'): digest(source / 'protocol.json'), str(Path(__file__)): digest(Path(__file__))})
    write_json(run / 'protocol.json', pp)
    a = copy.deepcopy(read(source / 'label_audit.json'))
    a['scope'] += ' Reused exact arrays and label audit; no label generation or filtering in budget control.'
    a['source_sha256'].update({str(source / 'label_audit.json'): digest(source / 'label_audit.json'),
                             str(run / 'protocol.json'): digest(run / 'protocol.json')})
    write_json(run / 'label_audit.json', a)
    qq = copy.deepcopy(q); qq['updates'] = 1400
    qq['evaluation'] = qq['evaluation'].replace('Final600', 'Final1400')
    qq['budget_control_parent'] = str(source)
    qq['source_sha256'].update({str(source / 'learning_protocol.json'): digest(source / 'learning_protocol.json'),
        str(source / 'learning_result.json'): digest(source / 'learning_result.json'),
        str(source / 'learning_audit.json'): digest(source / 'learning_audit.json'),
        str(run / 'protocol.json'): digest(run / 'protocol.json'), str(run / 'label_audit.json'): digest(run / 'label_audit.json'),
        str(run / 'inputs.npz'): digest(run / 'inputs.npz'), str(run / 'labels_offline_only.npz'): digest(run / 'labels_offline_only.npz'),
        str(Path(__file__)): digest(Path(__file__))})
    write_json(run / 'learning_protocol.json', qq)
    unchanged = {k:v for k,v in qq.items() if k not in ('updates','evaluation','budget_control_parent','source_sha256')}
    expected = {k:v for k,v in q.items() if k not in ('updates','evaluation','source_sha256')}
    assert unchanged == expected
    for name in ('inputs.npz','labels_offline_only.npz'):
        assert digest(run / name) == digest(source / name)
    write_json(run / 'budget_protocol.json', dict(status='frozen_before_budget_training', parent=str(source),
        original_updates=600, updates=1400, repeated_models=27,
        reason='Expanded expert training at600 has remaining orientation fit errors; use a fixed optimization-budget control before blaming held failure only on representation.',
        unchanged='All arrays, targets, PCA, full loss weights/masks, initial weight files, common prior, optimizer hyperparameters, architecture, held scenes, seeds and criteria identical. Fresh restart, no warm-start or best-checkpoint selection.',
        time_limit_seconds=600, source_sha256={str(run/'learning_protocol.json'):digest(run/'learning_protocol.json'),
            str(source/'learning_protocol.json'):digest(source/'learning_protocol.json'),str(Path(__file__)):digest(Path(__file__))}))
    (run / 'source_snapshot').mkdir()
    (run / 'source_snapshot/geometry_budget_control.py').write_bytes(Path(__file__).read_bytes())
    print('BUDGET_CONTROL_PREPARED', run, flush=True)


def summarize(run):
    from .audit_geometry_readout import finalize
    import numpy as np
    protocol = read(run/'budget_protocol.json'); verify(protocol)
    finalize(run)
    # The reusable audit's Markdown template has a literal600 title; label this
    # new, unsealed report with its actual preregistered budget, without edits to
    # any old code, protocols, predictions or quantitative audit results.
    report = run/'summary.md'; report.write_text(report.read_text().replace('final600', 'final1400', 1))
    current = read(run/'summary.json'); parent = Path(protocol['parent']); previous = read(parent/'summary.json')
    comparisons=[]
    for record in current['reports']:
        before=next(x for x in previous['reports'] if (x['scene'],x['condition'],x['seed'])==(record['scene'],record['condition'],record['seed']))
        comparisons.append(dict(scene=record['scene'],condition=record['condition'],seed=record['seed'],
                                before600=before,after1400=record))
    write_json(run/'budget_comparison.json',dict(status='complete',matched_models=27,reports=comparisons,
        scope='Matched fixed600vs1400final checkpoints, not checkpoint/seed selection. No action or task-success result.',
        source_sha256={str(run/'summary.json'):digest(run/'summary.json'),str(parent/'summary.json'):digest(parent/'summary.json'),
                       str(run/'budget_protocol.json'):digest(run/'budget_protocol.json')}))
    lines=['# S2h matched budget control: 600 vs1400','','|Scene|Condition|Functional cm600→1400|Face°600→1400|Root°600→1400|',
           '|---|---|---|---|---|']
    for scene in sorted({r['scene'] for r in comparisons}):
        for condition in read(run/'learning_protocol.json')['conditions']:
            values=[r for r in comparisons if r['scene']==scene and r['condition']==condition]
            nums=[]
            for k in ('functional_position_cm','face_angle_deg','root_angle_deg'):
                a=np.mean([v['before600']['held']['primary'][k] for v in values]);b=np.mean([v['after1400']['held']['primary'][k] for v in values])
                nums.append(f'{a:.2f}→{b:.2f}')
            lines.append('|'+scene+'|'+condition+'|'+'|'.join(nums)+'|')
    (run/'budget_comparison.md').write_text('\n'.join(lines)+'\n')
    print('BUDGET_COMPARISON_COMPLETE',run,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('path',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();(prepare if args.prepare else summarize)(args.path.resolve())
