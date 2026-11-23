"""Load all four completed frozen task indexes and build the audited common-start dataset."""
import json
from pathlib import Path
from .audit import digest
from .paths import ROOT,write_json
from .cartesian_pulse_batch import DEFAULT_PLAN,validate_plan
from .cartesian_pulse_dataset import build
from .audit_pulse_dataset import audit


def main():
    plan=json.loads(DEFAULT_PLAN.read_text());validate_plan(plan)
    expected={j['id']:j for j in plan['jobs']};seen=set();runs=[]
    for task in sorted({j['task'] for j in plan['jobs']}):
        index_path=ROOT/'outputs/cartesian_pulse_batches'/plan['batch_id']/task/'index.json'
        index=json.loads(index_path.read_text())
        if index['status']!='complete' or index['task']!=task or index['plan_sha256']!=digest(DEFAULT_PLAN):raise ValueError('Incomplete or changed task index: '+task)
        for row in index['records']:
            if row['id'] not in expected or row['id'] in seen:raise ValueError('Repeated or unexpected job')
            if any(row.get(k)!=v for k,v in expected[row['id']].items()):raise ValueError('Job fields changed')
            run=Path(row['run'])
            if digest(run/'manifest.json')!=row['manifest_sha256'] or digest(run/'result.json')!=row['result_sha256']:raise ValueError('Indexed run changed')
            manifest=json.loads((run/'manifest.json').read_text())
            if (manifest['task'],manifest['source_seed'],manifest['chunk'],manifest['pulse_start'])!=(row['task'],row['seed'],row['chunk'],row['pulse_start']):raise ValueError('Run identity changed')
            seen.add(row['id']);runs.append(run)
    if seen!=set(expected):raise ValueError('Missing frozen jobs')
    dataset=build(runs,ROOT/'configs/intervention_batch1_20260927.json');report=audit(dataset);write_json(dataset/'interface_audit.json',report)
    write_json(dataset/'batch_provenance.json',dict(plan=str(DEFAULT_PLAN),plan_sha256=digest(DEFAULT_PLAN),runs=len(runs)))
    print(json.dumps(dict(status='complete',dataset=str(dataset),audit=report),indent=2))

if __name__=='__main__':main()
