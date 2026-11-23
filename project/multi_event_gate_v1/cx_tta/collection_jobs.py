"""Frozen collection jobs, whole-scene identity and audited completion checks."""
import json
from pathlib import Path

from .audit import digest
from .paths import WORKSPACE_ROOT as ROOT, output_path

PROTOCOLS = {
    'stack_bowls_two': ROOT/'outputs/reports/protocols/stacking_training_20261013_094238_655283/protocol.json',
    'stack_blocks_two': ROOT/'outputs/reports/protocols/stacking_collection_v2_20261013_110328_874819/protocol.json',
}
ALIASES = {'bowls':'stack_bowls_two', 'blocks':'stack_blocks_two',
           **{k:k for k in PROTOCOLS}}
SPLIT_ALIASES = {'train':'adapter_fit', 'val':'adapter_validation', 'validation':'adapter_validation',
                 'adapter_fit':'adapter_fit', 'adapter_validation':'adapter_validation'}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def selected(value, aliases):
    values = list(dict.fromkeys(aliases.values())) if value == 'all' else value.split(',')
    try:
        normalized = [aliases[v] for v in values]
    except KeyError as exc:
        raise ValueError('Unknown task/split: '+str(exc)) from exc
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError('Select distinct tasks/splits; no duplicate scene jobs')
    return normalized


def collection_root(spec):
    return ROOT/'outputs/stacking_training/expert'/spec['task']/spec['split']


def validate_complete(run, spec, *, decode=False):
    """Exit zero alone is not success; verify dataset count, provenance and hashes."""
    run = output_path(Path(run))
    if run.parent != collection_root(spec).resolve():
        raise ValueError('Result outside expected task/split collection directory')
    manifest, result, audit = [read(run/name) for name in ('manifest.json','result.json','data_audit.json')]
    if (manifest.get('protocol_sha256') != spec['protocol_sha256']
        or manifest.get('task') != spec['task'] or manifest.get('split') != spec['split']
        or result.get('status') != 'collection_complete'
        or result.get('accepted_episodes') != spec['episodes']
        or result.get('task') != spec['task'] or result.get('split') != spec['split']
        or audit.get('passed') is not True or audit.get('scenes') != spec['episodes']
        or audit.get('task') != spec['task'] or audit.get('split') != spec['split']):
        raise ValueError('Incomplete or mismatched collection: '+str(run))
    records = read(run/'attempts.json')
    rows = [r for r in records if r.get('accepted') is True]
    if len(rows) != spec['episodes'] or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Incorrect or duplicate accepted scenes')
    total_bytes = 0
    for r in rows:
        if not spec['seed_window'][0] <= r['seed'] < spec['seed_window'][1]:
            raise ValueError('Scene outside registered split')
        if (r.get('task') != spec['task'] or r.get('split') != spec['split']
            or r.get('plan_success') is not True or r.get('replay_success') is not True):
            raise ValueError('Unqualified expert scene')
        path = output_path(Path(r['hdf5']))
        if not path.is_relative_to(run) or digest(path) != r['capture']['source_sha256']:
            raise ValueError('Changed capture or escaped path')
        total_bytes += path.stat().st_size
    if decode:
        from .stacking_training_data import audit_collection
        audit_collection(run, Path(spec['protocol']))
    return dict(run=str(run), task=spec['task'], split=spec['split'], episodes=len(rows),
                seeds=[r['seed'] for r in rows], hdf5_bytes=total_bytes,
                protocol_sha256=spec['protocol_sha256'])


def inventory(spec, *, retry_incomplete=False, resume=False):
    complete, incomplete = [], []
    for run in sorted(collection_root(spec).glob('collect_*')):
        if not (run/'manifest.json').is_file():
            continue
        if (run/'data_audit.json').is_file() and not (run/'failure.json').exists():
            complete.append(validate_complete(run, spec))
        else:
            incomplete.append(str(run))
    if complete:
        # Deterministically reuse the earliest complete audited collection. Never
        # concatenate repeated runs or count repeated scenes as extra examples.
        return dict(reuse=complete[0], duplicates=complete[1:], incomplete=incomplete)
    if incomplete and resume:
        from .collection_resume import resume_state
        if len(incomplete) != 1:
            raise ValueError('Multiple incomplete collections; select the authoritative run before resuming; no automatic merge')
        state = resume_state(Path(incomplete[0]),spec)
        return dict(reuse=None,duplicates=[],incomplete=incomplete,resume=state)
    if incomplete and not retry_incomplete:
        raise ValueError('Existing active/incomplete collection; do not duplicate it. Inspect '+
                         incomplete[0]+'; --retry-incomplete is only for jobs confirmed stopped.')
    return dict(reuse=None, duplicates=[], incomplete=incomplete)


def build_jobs(args):
    from .collect_stacking_experts import options, preflight as v1
    from .collect_stacking_experts_v2 import preflight as v2
    from .stacking_training_design import load_design
    from .stacking_training_data import audit_collection
    tasks, splits = selected(args.tasks, ALIASES), selected(args.splits, SPLIT_ALIASES)
    templates, evidence = {}, {}
    for task in tasks:
        protocol = PROTOCOLS[task]
        p = load_design(protocol)
        revision = 'v2' if 'collector_revision' in p else 'v1'
        preflight = v2 if revision == 'v2' else v1
        smoke = ROOT/'outputs/stacking_training/expert'/task/'smoke'
        verified = []
        for run in sorted(smoke.glob('collect_*')):
            m = read(run/'manifest.json')
            if m.get('protocol_sha256') == digest(protocol) and (run/'data_audit.json').is_file():
                verified.append(dict(run=str(run), audit=audit_collection(run, protocol)))
        if not verified:
            raise ValueError('Real successful smoke is required for '+task+'/'+revision)
        templates[task] = (p, protocol, revision, preflight)
        evidence[task] = dict(smoke=verified, preflights={})
    jobs = []
    # Both long fitting jobs first, then independent validation jobs.
    for split in splits:
        for task in tasks:
            p, protocol, revision, preflight = templates[task]
            a = options(['--protocol', str(protocol), '--task',task,'--split',split,'--gpu',str(args.gpu),'--dry-run'])
            manifest = preflight(a)[3]
            evidence[task]['preflights'][split] = manifest
            spec = dict(id=task+'_'+split, task=task, split=split, episodes=p['splits'][split]['episodes'],
                seed_window=manifest['seed_window'], protocol=str(protocol), protocol_sha256=digest(protocol),
                collector_version=revision, min_free_gb=args.min_disk_gb,
                retry_incomplete=args.retry_incomplete)
            spec['existing'] = inventory(spec, resume=True)
            jobs.append(spec)
    return jobs, evidence
