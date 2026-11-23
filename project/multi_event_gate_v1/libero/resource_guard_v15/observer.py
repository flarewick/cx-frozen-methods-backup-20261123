"""CPU-only account/fleet observer: raw components, process birth identities,3s cadence."""
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

from policy import L
sys.path.insert(0, str(L.parents[1]/'deployment'))
from gpu_policy import inventory as physical_inventory
sys.path.insert(0, str(L / 'official_baseline_v14'))
from common import write, append

STATE = L / 'outputs/resource_observer_v15.json'
HISTORY = L / 'outputs/resource_observer_v15_window.json'


def parse_pmon(text):
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        values = [int(x) if x.isdigit() else None for x in parts[3:5]]
        assert all(x is None or 0 <= x <= 100 for x in values)
        rows[(int(parts[0]), int(parts[1]))] = values
    return rows


def proc(pid, uptime, now, previous):
    p = Path('/proc', str(pid))
    st = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    hz = os.sysconf('SC_CLK_TCK')
    birth = int(st[19]); ticks = int(st[11]) + int(st[12])
    old = previous.get((pid, birth)); previous[(pid, birth)] = (now, ticks)
    cpu = 0. if old is None else 100 * (ticks - old[1]) / hz / max(now - old[0], .001)
    return dict(pid=pid, birth=birth, uid=p.stat().st_uid, age=max(0, uptime - birth / hz),
                cpu=cpu, rss=int(st[21]) * os.sysconf('SC_PAGE_SIZE'))


def sample(previous):
    cmd = lambda args: subprocess.run(['nvidia-smi', *args], check=True, capture_output=True,
                                     text=True, timeout=4).stdout
    components = parse_pmon(cmd(['pmon', '-s', 'u', '-c', '1']))
    xml = ET.fromstring(cmd(['-q', '-x']))
    now = time.monotonic(); wall = time.time(); uptime = float(Path('/proc/uptime').read_text().split()[0])
    cards = []
    for i, g in enumerate(xml.findall('gpu')):
        number = lambda path: float(g.findtext(path).split()[0])
        ps = []
        for node in g.findall('./processes/process_info'):
            try:
                pid = int(node.findtext('pid')); p = proc(pid, uptime, now, previous)
            except (FileNotFoundError, ProcessLookupError):
                continue
            sm, bd = components.get((i, pid), (None, None))
            token = node.findtext('used_memory', '').split()
            memory = float(token[0]) if token and token[0].replace('.', '', 1).isdigit() else 0.
            p.update(sm=sm, bandwidth=bd, memory_mib=memory,
                     metrics_available=(sm is not None and bd is not None and memory > 0))
            ps.append(p)
        mem = number('./fb_memory_usage/used'); sm = number('./utilization/gpu_util')
        cards.append(dict(index=i, uuid=g.findtext('uuid'), sm=sm,
                          bandwidth=number('./utilization/memory_util'), memory_mib=mem,
                          total_mib=number('./fb_memory_usage/total'), processes=ps,
                          pids=[p['pid'] for p in ps], own_pids=[p['pid'] for p in ps if p['uid'] == os.getuid()],
                          available=not g.findall('./processes/process_info') and mem <= 64 and sm <= 5))
    return dict(monotonic=now, wall_time=wall, observer_pid=os.getpid(), cards=cards)


def means(rows, now, *, total_mib=None):
    """Arithmetic mean of raw components THEN MAX; missing samples count zero."""
    assert rows
    n = len(rows); duration = now - rows[0]['monotonic']
    sm = sum(r.get('sm') or 0 for r in rows) / n
    bd = sum(r.get('bandwidth') or 0 for r in rows) / n
    valid = [r['monotonic'] for r in rows if r.get('metrics_available')]
    gap = now - valid[-1] if valid else 999.
    maxgap = max([b['monotonic'] - a['monotonic'] for a, b in zip(rows, rows[1:])] or [0.])
    result = dict(sm=sm, bandwidth=bd, util=max(sm, bd),
                  memory_mib=sum(r['memory_mib'] for r in rows) / n,
                  cpu=sum(r.get('cpu', 0.) for r in rows) / n,
                  rss_max=max(r.get('rss', 0) for r in rows), samples=n,
                  coverage_seconds=duration, metric_gap_seconds=gap,
                  full_window=duration >= 174 and n >= 50 and maxgap <= 9,
                  missing_samples_counted_zero=True)
    if total_mib:
        result['memory_pct'] = 100 * result['memory_mib'] / total_mib
    return result


def aggregate(current, history):
    now = current['monotonic']
    for c in current['cards']:
        historical = [row['cards'][c['index']] for row in history]
        for p in c['processes']:
            identity = (p['pid'], p['birth']); rows = []
            for h, card in zip(history, historical):
                same = next((v for v in card['processes'] if (v['pid'], v['birth']) == identity), None)
                if same:
                    rows.append(dict(same, monotonic=h['monotonic']))
            p['rolling180'] = means(rows, now)
            p['recent60'] = means([r for r in rows if r['monotonic'] >= now - 60], now)
        account = []
        for h, card in zip(history, historical):
            owned = [p for p in card['processes'] if p['uid'] == os.getuid()]
            # Device components cap process sums so foreign load is never credited.
            account.append(dict(monotonic=h['monotonic'],
                                sm=min(card['sm'], sum(p['sm'] or 0 for p in owned)),
                                bandwidth=min(card['bandwidth'], sum(p['bandwidth'] or 0 for p in owned)),
                                memory_mib=sum(p['memory_mib'] for p in owned),
                                metrics_available=bool(owned) and all(p['metrics_available'] for p in owned)))
        c['own_rolling180'] = means(account, now, total_mib=c['total_mib'])
        device = [dict(monotonic=h['monotonic'], sm=card['sm'], bandwidth=card['bandwidth'],
                       memory_mib=card['memory_mib'], metrics_available=True)
                  for h, card in zip(history, historical)]
        c['device_rolling180'] = means(device, now)
        c['device_recent60'] = means([r for r in device if r['monotonic'] >= now - 60], now)
    current.update(available_indices=[c['index'] for c in current['cards'] if c['available']],
                   own_gpu_count=sum(bool(c['own_pids']) for c in current['cards']), max_gpus=3,
                   observer_pid=os.getpid(), schema='governor180_v15')
    return current


def latest():
    value = json.loads(STATE.read_text())
    if value.get('error') or time.time() - value['wall_time'] > 8:
        raise RuntimeError('Fleet observer stale or failed: ' + str(value.get('error')))
    return value


def admission_snapshot():
    """Recheck live occupation; never admit from a historical empty-card sample."""
    observed = latest(); fresh = physical_inventory()
    for c in observed['cards']:
        current = next(v for v in fresh['cards'] if v['uuid'] == c['uuid'])
        if set(c['pids']) != set(current['pids']) or set(c['own_pids']) != set(current['own_pids']):
            raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
        for p in c['processes']:
            try:
                birth = int(Path('/proc',str(p['pid']),'stat').read_text().rsplit(')',1)[1].split()[19])
            except (FileNotFoundError,ProcessLookupError):
                raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
            if birth != p['birth']:
                raise RuntimeError('Fleet changed since180s observation; wait for fresh observer')
        c['available'] = current['available']
    observed['available_indices'] = [c['index']for c in observed['cards']if c['available']]
    observed['wall_time'] = time.time()
    return observed


def main():
    lock = (L / 'outputs/resource_observer_v15.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = deque(); previous = {}; start = time.monotonic()
    while time.monotonic() - start < 86400:
        begin = time.monotonic()
        try:
            raw = sample(previous)
            history.append(raw)
            while history and history[0]['monotonic'] < raw['monotonic'] - 180:
                history.popleft()
            # JSON copy keeps raw observations independent of derived fields.
            current = aggregate(json.loads(json.dumps(raw)), history)
            write(STATE, current); write(HISTORY, list(history))
            append(L / 'outputs/resource_observer_v15.jsonl', raw)
        except Exception as exc:
            write(STATE, dict(wall_time=time.time(), error=repr(exc), observer_pid=os.getpid()))
        time.sleep(max(.1, 3 - (time.monotonic() - begin)))


if __name__ == '__main__':
    main()
