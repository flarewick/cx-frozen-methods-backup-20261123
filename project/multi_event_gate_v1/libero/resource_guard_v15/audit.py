"""Independently recompute current-rule statistics from raw fleet observations."""
import json
import math
from collections import deque
from pathlib import Path
from policy import L, evaluate


def verify_attempt(attempt):
    at = Path(attempt)
    rows = [json.loads(x) for x in (at/'resources.jsonl').read_text().splitlines()]
    assert rows[-1]['own_gpu_released']
    samples = {r['observer_monotonic']:r for r in rows if r.get('schema')=='governor180_v15'}
    history = deque(); checked = 0; precheck_stops = []; observer = None
    with (L/'outputs/resource_observer_v15.jsonl').open() as stream:
        for line in stream:
            raw = json.loads(line); now = raw['monotonic']
            if raw.get('observer_pid') != observer:
                history.clear(); observer=raw.get('observer_pid')
            history.append(raw)
            while history and history[0]['monotonic'] < now-180:
                history.popleft()
            if now not in samples:
                continue
            s = samples[now]; fleet = s['fleet'];assert fleet['observer_pid']==observer
            identities = {(p['pid'],p['birth']) for p in s['process_identities']}
            assert evaluate(fleet, identities) == s['server_verdict']
            for c in fleet['cards']:
                for p in c['processes']:
                    identity = (p['pid'],p['birth']); actual = []
                    for h in history:
                        actual += [q for q in h['cards'][c['index']]['processes']
                                   if (q['pid'],q['birth'])==identity]
                    assert actual
                    n=len(actual); stats=p['rolling180']
                    expected=dict(sm=sum(q['sm']or 0 for q in actual)/n,
                                  bandwidth=sum(q['bandwidth']or 0 for q in actual)/n,
                                  cpu=sum(q['cpu']for q in actual)/n,
                                  memory_mib=sum(q['memory_mib']for q in actual)/n,
                                  rss_max=max(q['rss']for q in actual))
                    expected['util']=max(expected['sm'],expected['bandwidth'])
                    for k,v in expected.items():
                        assert math.isclose(stats[k],v,abs_tol=1e-8), (p['pid'],k)
                # Independent account-card aggregation, including zero/missing samples.
                sm=[];bd=[];memory=[]
                for h in history:
                    raw_card=h['cards'][c['index']]
                    owned=[p for p in raw_card['processes']if p['uid']==__import__('os').getuid()]
                    sm.append(min(raw_card['sm'],sum(p['sm']or 0 for p in owned)))
                    bd.append(min(raw_card['bandwidth'],sum(p['bandwidth']or 0 for p in owned)))
                    memory.append(sum(p['memory_mib']for p in owned))
                expected_util=max(sum(sm)/len(sm),sum(bd)/len(bd))
                assert math.isclose(c['own_rolling180']['util'],expected_util,abs_tol=1e-8)
                assert math.isclose(c['own_rolling180']['memory_pct'],100*sum(memory)/len(memory)/c['total_mib'],abs_tol=1e-8)
            if not s['server_verdict']['passed']:
                assert now==max(samples),'Violation must be the terminal resource observation'
                assert any('stopped'in v for v in rows)
                assert rows[-1]['cleanup_seconds_after_start']-s['seconds']<6
                precheck_stops.append(s['server_verdict']['reasons'])
            checked+=1
    assert checked==len(samples) and checked
    return dict(passed=True,resource_schema='governor180_v15',samples=checked,
                raw_component_statistics_recomputed=True,pregrace_protective_stops=precheck_stops,
                GPU_released=True)
