"""Frozen scene-grouped intervention jobs with audited, per-task resumption."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from .audit import digest
from .audit_interventions import audit_run
from .paths import ROOT, output_path, replace_json, write_json
from .probe_action_replay import load_source, source_runtime

DEFAULT_PLAN = ROOT / 'configs/intervention_batch1_20260927.json'


def temporal_chunks(full_chunks):
    """Three temporal strata, independent of GT movement or branch effects."""
    if not full_chunks:
        raise ValueError('No full action chunks')
    return sorted({full_chunks[0], full_chunks[(len(full_chunks) - 1) // 2], full_chunks[-1]})


def validate_plan(plan):
    if plan['schema_version'] != 1 or not plan['batch_id'].isidentifier():
        raise ValueError('Invalid batch plan')
    for name, expected in plan['code_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen collection code changed: ' + name)
    ids, groups = set(), {}
    for job in plan['jobs']:
        if not job['id'].isidentifier() or job['id'] in ids:
            raise ValueError('Invalid or duplicate job ID')
        ids.add(job['id'])
        group = (job['task'], job['seed'])
        if groups.setdefault(group, job['split']) != job['split']:
            raise ValueError('Scene leakage across splits')
        if job['split'] not in ('train', 'validation'):
            raise ValueError('Unknown split')
    return True


def validate_sources(jobs):
    cached = {}
    for job in jobs:
        key = (job['source'], job['seed'])
        if key not in cached:
            p = Path(job['source'])
            source_runtime(p)
            cached[key] = load_source(p, job['seed'])
        manifest, episode, rows = cached[key]
        if (manifest['args']['task'] != job['task']
                or rows[job['chunk']]['executed'] != 32
                or digest(Path(job['source']) / 'traces' / f"r0_seed{job['seed']}.jsonl") != job['trace_sha256']):
            raise ValueError('Frozen source changed: ' + job['id'])


def completed_attempt(attempts, job, plan):
    """Only reuse a fully audited matching attempt; never hide damaged data."""
    if not attempts.exists():
        return None
    for run in sorted(attempts.iterdir()):
        if not (run / 'result.json').exists() or (run / 'failure.json').exists():
            continue
        m = json.loads((run / 'manifest.json').read_text())
        if (m['source'] != job['source'] or m['source_seed'] != job['seed']
                or m['chunk'] != job['chunk'] or m['proposal_set'] != 'shared_v1'
                or m['probe_strength'] != plan['probe_strength']
                or m['collector_sha256'] != plan['code_hashes']['cx_tta/collect_interventions.py']
                or m['contract_sha256'] != plan['code_hashes']['cx_tta/intervention_contract.py']):
            raise ValueError('Completed attempt provenance differs: ' + str(run))
        audit_run(run)
        return run
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--task', required=True,
                        choices=('beat_block_hammer', 'place_shoe', 'turn_switch', 'place_empty_cup'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('GPU must be a nonnegative index')
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    jobs = [j for j in plan['jobs'] if j['task'] == args.task]
    if not jobs:
        parser.error('No jobs for task')
    validate_sources(jobs)
    print(json.dumps({'task': args.task, 'checkpoints': len(jobs), 'branches': 7 * len(jobs),
                      'scenes': len({j['seed'] for j in jobs}),
                      'train_checkpoints': sum(j['split'] == 'train' for j in jobs),
                      'validation_checkpoints': sum(j['split'] == 'validation' for j in jobs),
                      'coordinate_coverage': sorted({(j['seed'] + j['chunk']) % 14 for j in jobs}),
                      'dry_run': args.dry_run}), flush=True)
    if args.dry_run:
        return
    dest = output_path(ROOT / 'outputs/intervention_batches' / plan['batch_id'] / args.task)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This task batch is already running')
        frozen = dest / 'plan.json'
        if frozen.exists():
            if json.loads(frozen.read_text()) != plan:
                raise ValueError('Batch plan differs from existing output')
        else:
            write_json(frozen, plan)
        records = []
        for job in jobs:
            attempts = dest / 'jobs' / job['id'] / 'attempts'
            if job.get('reuse_run'):
                run = output_path(Path(job['reuse_run']))
                if digest(run / 'manifest.json') != job['reuse_manifest_sha256']:
                    raise ValueError('Frozen reuse manifest changed')
                m = json.loads((run / 'manifest.json').read_text())
                if (m['source'] != job['source'] or m['source_seed'] != job['seed']
                        or m['chunk'] != job['chunk'] or m['proposal_set'] != 'shared_v1'
                        or m['probe_strength'] != plan['probe_strength']
                        or m['contract_sha256'] != plan['code_hashes']['cx_tta/intervention_contract.py']):
                    raise ValueError('Frozen reuse source/proposals differ')
                audit_run(run)
            else:
                run = completed_attempt(attempts, job, plan)
            reused = run is not None
            if run is None:
                attempts.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                run = attempts / ('attempt_' + stamp)
                command = ['bash', str(ROOT / 'scripts/collect_interventions.sh'),
                           '--source', job['source'], '--seed', str(job['seed']),
                           '--chunk', str(job['chunk']), '--proposal-set', 'shared_v1',
                           '--probe-strength', str(plan['probe_strength']), '--gpu', args.gpu,
                           '--run-dir', str(run)]
                log = attempts / ('attempt_' + stamp + '.log')
                print(f"START {job['id']} split={job['split']} log={log}", flush=True)
                with log.open('x') as stream:
                    status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
                if status:
                    replace_json(dest / 'progress.json', {'status': 'failed', 'completed': records,
                                 'failed_job': job, 'returncode': status, 'log': str(log)})
                    raise RuntimeError('Collection failed; rerun same command to resume. See ' + str(log))
                audit_run(run)
            records.append({**job, 'run': str(run), 'reused': reused})
            replace_json(dest / 'progress.json', {'status': 'running', 'completed': records,
                                                 'total_jobs': len(jobs)})
            print(f"PASS {job['id']} ({len(records)}/{len(jobs)}), reused={reused}", flush=True)
        replace_json(dest / 'index.json', {'status': 'complete', 'plan_sha256': digest(args.plan),
                     'task': args.task, 'records': records,
                     'scope': 'grouped intervention data; not task success-rate evaluation'})
        replace_json(dest / 'progress.json', {'status': 'complete', 'completed': records,
                                             'total_jobs': len(jobs)})
        print(f"COMPLETE {args.task}: {len(records)}/{len(jobs)} checkpoints, "
              f"{7 * len(records)} branches. Index: {dest / 'index.json'}", flush=True)


if __name__ == '__main__':
    main()
