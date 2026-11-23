"""Exact LIBERO v15 statistics, copied without sampling or sibling writes."""
import os
def parse_pmon(text):
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or (not parts[1].isdigit()):
            continue
        values = [int(x) if x.isdigit() else None for x in parts[3:5]]
        assert all((x is None or 0 <= x <= 100 for x in values))
        rows[int(parts[0]), int(parts[1])] = values
    return rows

def means(rows, now, *, total_mib=None):
    """Arithmetic mean of raw components THEN MAX; missing samples count zero."""
    assert rows
    n = len(rows)
    duration = now - rows[0]['monotonic']
    sm = sum((r.get('sm') or 0 for r in rows)) / n
    bd = sum((r.get('bandwidth') or 0 for r in rows)) / n
    valid = [r['monotonic'] for r in rows if r.get('metrics_available')]
    gap = now - valid[-1] if valid else 999.0
    maxgap = max([b['monotonic'] - a['monotonic'] for a, b in zip(rows, rows[1:])] or [0.0])
    result = dict(sm=sm, bandwidth=bd, util=max(sm, bd), memory_mib=sum((r['memory_mib'] for r in rows)) / n, cpu=sum((r.get('cpu', 0.0) for r in rows)) / n, rss_max=max((r.get('rss', 0) for r in rows)), samples=n, coverage_seconds=duration, metric_gap_seconds=gap, full_window=duration >= 174 and n >= 50 and (maxgap <= 9), missing_samples_counted_zero=True)
    if total_mib:
        result['memory_pct'] = 100 * result['memory_mib'] / total_mib
    return result

def aggregate(current, history):
    now = current['monotonic']
    for c in current['cards']:
        historical = [row['cards'][c['index']] for row in history]
        for p in c['processes']:
            identity = (p['pid'], p['birth'])
            rows = []
            for h, card in zip(history, historical):
                same = next((v for v in card['processes'] if (v['pid'], v['birth']) == identity), None)
                if same:
                    rows.append(dict(same, monotonic=h['monotonic']))
            p['rolling180'] = means(rows, now)
            p['recent60'] = means([r for r in rows if r['monotonic'] >= now - 60], now)
        account = []
        for h, card in zip(history, historical):
            owned = [p for p in card['processes'] if p['uid'] == os.getuid()]
            account.append(dict(monotonic=h['monotonic'], sm=sum((p['sm'] or 0 for p in owned)), bandwidth=sum((p['bandwidth'] or 0 for p in owned)), memory_mib=sum((p['memory_mib'] for p in owned)), metrics_available=bool(owned) and all((p['metrics_available'] for p in owned))))
        device = [dict(monotonic=h['monotonic'], sm=card['sm'], bandwidth=card['bandwidth'], memory_mib=card['memory_mib'], metrics_available=True) for h, card in zip(history, historical)]
        c['device_rolling180'] = means(device, now)
        own = means(account, now, total_mib=c['total_mib'])
        for component in ('sm', 'bandwidth'):
            own[component] = min(100., own[component], c['device_rolling180'][component])
        own['util'] = max(own['sm'], own['bandwidth'])
        c['own_rolling180'] = own
        c['device_recent60'] = means([r for r in device if r['monotonic'] >= now - 60], now)
    current.update(available_indices=[c['index'] for c in current['cards'] if c['available']], own_gpu_count=sum((bool(c['own_pids']) for c in current['cards'])), max_gpus=3, observer_pid=os.getpid(), schema='governor180_v15')
    return current
