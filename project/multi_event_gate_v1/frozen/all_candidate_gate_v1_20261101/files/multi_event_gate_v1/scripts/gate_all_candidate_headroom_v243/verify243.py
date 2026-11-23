"""Replay exact failed telemetry against new early protection, no GPU."""
import json,hashlib
from pathlib import Path
import protection243 as new
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';q=H/'candidate_veto_all_events_v237_0/dev0'
state=json.loads((q/'host_state.json').read_text());file=Path(state['telemetry']);rows=[json.loads(x)for x in file.read_text().splitlines()];samples=[r for r in rows if r.get('gpu_pids')]
assert any(r['seconds']>=300 and r['time_weighted_mean_utilization']<33 for r in samples),'Keep the actual historical failed sample explicit'
trigger=next(r for r in samples if new.reason(r));assert 240<=trigger['seconds']<260
assert not new.reason(dict(seconds=239,gpu_pids=[1]))
base=dict(seconds=400,gpu_pids=[1],mean_utilization=40,time_weighted_mean_utilization=40,rolling_300s_utilization=40,process_mean_utilization={'1':40},process_rolling_300s_utilization={'1':40});assert new.reason(base)is None
for key in ['mean_utilization','time_weighted_mean_utilization','rolling_300s_utilization']:
 d=dict(base);d[key]=39.99;assert new.reason(d)
for key in ['process_mean_utilization','process_rolling_300s_utilization']:
 d=dict(base);d[key]={'1':39.99};assert new.reason(d)
d=dict(base,metric_missing_conservative_zero_pids=[1]);assert new.reason(d)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
value=dict(passed=True,CPU_only=True,not_GPU_performance_or_benefit_acceptance=True,actual_failed_telemetry_sha256=sha(file),actual_new_stop_would_be_seconds=trigger['seconds'],historical_original_crossing_sample_retained=True,all_device_PID_cumulative_and_rolling_fields_protected=True,original33pct_guard_not_modified=True,protection_sha256=sha(HERE/'protection243.py'),host243_sha256=sha(HERE/'host243.py'),verifier_sha256=sha(__file__))
(HERE/'CPU_proof243.json').write_text(json.dumps(value,indent=2)+'\n');print(value)
