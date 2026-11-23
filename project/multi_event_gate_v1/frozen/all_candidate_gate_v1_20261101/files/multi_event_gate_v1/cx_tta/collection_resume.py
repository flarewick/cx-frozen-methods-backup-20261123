"""Resume one collection at validated episode boundaries, never splice trajectories."""
import os
import socket
from pathlib import Path

from .audit import digest
from .paths import output_path
from .collection_jobs import read, collection_root

EXCLUDED = {'expert_plan_failed', 'expert_replay_failed',
            'expert_missing_grasp_target', 'unstable_scene'}


def resume_state(run, spec, *, decode=False):
    run = output_path(Path(run))
    manifest = read(run/'manifest.json')
    if (run.parent != collection_root(spec).resolve()
        or manifest.get('task') != spec['task'] or manifest.get('split') != spec['split']
        or manifest.get('protocol_sha256') != spec['protocol_sha256']
        or manifest.get('seed_window') != spec['seed_window']
        or manifest.get('episodes') != spec['episodes']):
        raise ValueError('Resume task/split/protocol/count mismatch: '+str(run))
    rows = read(run/'attempts.json') if (run/'attempts.json').exists() else []
    if not isinstance(rows, list):
        raise ValueError('Invalid attempt journal')
    accepted, terminal, bytes_used = [], set(), 0
    for row in rows:
        seed = row.get('seed')
        if (type(seed) is not int or not spec['seed_window'][0] <= seed < spec['seed_window'][1]
            or row.get('task') != spec['task'] or row.get('split') != spec['split']):
            raise ValueError('Attempt outside registered collection')
        if row.get('accepted') is True:
            if (seed in terminal or row.get('plan_success') is not True
                or row.get('replay_success') is not True or row.get('episode_index') != len(accepted)):
                raise ValueError('Duplicate/unqualified/misindexed accepted scene')
            path = output_path(Path(row['hdf5']))
            if not path.is_relative_to(run) or digest(path) != row['capture']['source_sha256']:
                raise ValueError('Accepted capture changed or escaped its collection; refusing silent replacement')
            if decode:
                from .stacking_training_data import qualify_capture
                qualify_capture(path, spec['task'], seed, spec['split'], spec['protocol_sha256'])
            accepted.append(row); terminal.add(seed); bytes_used += path.stat().st_size
        elif row.get('excluded_reason') in EXCLUDED:
            if seed in terminal:
                raise ValueError('Repeated terminal scene in journal')
            terminal.add(seed)
        # Unknown infrastructure failures and uncommitted episodes are retried
        # at the same seed, never counted as scene exclusions or successes.
    if len(accepted) > spec['episodes']:
        raise ValueError('More accepted scenes than the frozen target')
    remaining = [s for s in range(*spec['seed_window']) if s not in terminal]
    return dict(run=str(run), accepted=len(accepted), requested=spec['episodes'],
                remaining=spec['episodes']-len(accepted), attempts=len(rows),
                next_seed=remaining[0] if remaining else None,
                terminal_seeds=sorted(terminal), hdf5_bytes=bytes_used)


def local_process_identity():
    # boot_id and process start tick protect against PID reuse on this host.
    return dict(host=socket.gethostname(), pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_tick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])


def check_unmanaged_writer(run):
    """Old direct collectors did not acquire scene locks; reject detectable live ones."""
    run = str(output_path(Path(run)))
    identity = read(Path(run)/'writer.json')
    current = local_process_identity()
    if identity.get('host') == current['host'] and identity.get('boot_id') == current['boot_id']:
        path = Path('/proc',str(identity.get('pid')),'stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[19] == identity.get('start_tick'):
            if identity.get('pid') != os.getpid():
                raise RuntimeError('Collection still has a live writer: '+run)
    # Detect earlier direct scripts through their open console.log. Managed
    # processes, including other hosts, are also protected by the scene flock.
    target = run+'/console.log'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc/'fd').iterdir():
                if os.readlink(fd) != target:
                    continue
                info = (proc/'fdinfo'/fd.name).read_text()
                flags = next(line.split()[1] for line in info.splitlines() if line.startswith('flags:'))
                if int(flags,8) & os.O_ACCMODE != os.O_RDONLY:
                    raise RuntimeError('An old direct collector still writes '+target)
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
