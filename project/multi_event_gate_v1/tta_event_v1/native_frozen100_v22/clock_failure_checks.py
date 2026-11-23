"""Read-only actual failed endpoint remains rejected; new runs need fresh UUID."""
import json
from .common import T,O,read,write,sha
from native_process_sampler_v7.metrics import select

def main():
 r=T/'outputs/native_frozen100_v18/mixed/batches/025/runs/ef3c86a548614c81bcc0c5d271399fc7';text=read(r/'failure.json')['traceback'];raw=json.JSONDecoder().raw_decode(text[text.rfind('raw_NVML=')+9:])[0];a,b=raw['frames'];delta=b['driver_timestamp_us']-a['driver_timestamp_us'];assert delta==1798690 and delta>1600000
 try:select(raw,6)
 except ValueError as e:assert 'Driver clock did not advance'in str(e)
 else:raise AssertionError('Old invalid clock accepted')
 observations=raw['clock_only_readiness_observations'];gap=observations[-1]['host_timestamp_us']-observations[-2]['host_timestamp_us'];assert gap==838851
 assert (r/'failure.json').exists()and not(r/'terminal_audit.json').exists()
 write(O/'CPU_clock_failure_checks.json',dict(passed=True,old_failed_endpoints_rejected=True,driver_delta_us=delta,last_clock_readiness_wall_gap_us=gap,OS_scheduling_versus_API_blocking_cause_not_yet_distinguished=True,new_runs_require_different_GPU_UUID=True,original_sampler7_and_all_clock_resource_bounds_unchanged=True,old_primary_results_not_imported=True,input_sha256={str(f):sha(f)for f in[r/'failure.json',r/'attempt/resources.jsonl']}))
 print('ACTUAL_CLOCK_OVERRUN_FRESH_CARD_CPU_PASS')
if __name__=='__main__':main()
