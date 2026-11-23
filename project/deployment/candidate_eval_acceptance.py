"""Read-only first sustained-run receipt; original guard continues independently."""
import argparse,json,os,zlib
from pathlib import Path
from candidate_eval_status import status,P
from evaluate import host
from gpu_policy import inventory

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',type=Path);args=ap.parse_args()
    live=status()
    active=json.loads((P/'target_candidate_eval_active_20261025.json').read_text())
    state=json.loads(Path(active['status_file']).read_text())
    ready=json.loads((Path(state['barrier'])/'ready.json').read_text())
    previous={j['condition']:set(j['previous_sessions']) for j in ready['plan']['jobs']}
    def current_session(file,root,condition):
        return file.relative_to(root/'sessions').parts[0] not in previous[condition]
    if not live['master_running'] or live['phase']!='running' or live['seconds']<350:raise RuntimeError('Need live350s resource evidence')
    rows=[json.loads(line) for line in Path(live['telemetry']).read_text().splitlines() if line.strip()]
    eligible=[r for r in rows if r.get('seconds',0)>=280 and r.get('gpu_pids')]
    if not eligible:raise RuntimeError('No post280 resource samples')
    for r in eligible:
        if r['account_gpu_count']>3 or len(r['gpu_pids'])!=1 or r['memory_mib']>24563:raise RuntimeError('Card/account constraint failed')
        if r['time_weighted_mean_utilization']<33:raise RuntimeError('Device cumulative threshold failed')
        for pid in r['gpu_pids']:
            key=str(pid)
            for field in ['process_mean_utilization','process_rolling_300s_utilization']:
                if r[field][key]<33:raise RuntimeError('Process utilization threshold failed')
            if not 4096<=r['process_memory_mib'][key]<=24563 or not 40<=r['cpu'][key]<=1600 or r['rss_bytes'][key]>128*1024**3:raise RuntimeError('PID resources failed')
        if r['seconds']>=300 and r['rolling_300s_utilization']<33:raise RuntimeError('Device rolling threshold failed')
    checks={};parity=[]
    for condition,job in live['conditions'].items():
        root=Path(job['run']);checked=0;active=0;queries=0
        m=json.loads((root/'manifest.json').read_text());alpha=m['candidate']['alpha']
        for f in(root/'sessions').glob('process_*/queries/*.jsonl.gz'):
            if not current_session(f,root,condition):continue
            data=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(f.read_bytes()).decode('utf-8')
            for line in data.split("\n")[:-1]:
                if not line.strip():continue
                q=json.loads(line);queries+=1
                if q['evaluated_condition']!=condition or q['candidate_alpha']!=alpha:raise RuntimeError('Candidate trace configuration mismatch')
                checked+=bool(q['reference_baseline_prefix_checked']);active+=bool(q['multi_event']['active'])
                if q['max_abs_correction']>.100001*alpha or q['gripper_strength']!=0.:raise RuntimeError('Candidate scale/GRIP trace mismatch')
        if checked<1 or active<1:raise RuntimeError('Need real old-prefix comparisons and active interventions for both conditions')
        checks[condition]=dict(queries=queries,original_prefix_checks=checked,active_interventions=active,alpha=alpha,completed=job['completed'],successes=job['successes'],checkpoint_sha256=m['checkpoint_sha256'])
        for f in(root/'sessions').glob('process_*/deployment_query_parity.json'):
            if not current_session(f,root,condition):continue
            x=json.loads(f.read_text())
            if x.get('passed'):parity.append(x)
    if not parity or not any(all(x.get('queries_by_condition',{}).get(c,0)>=3 for c in live['conditions']) for x in parity):raise RuntimeError('Both real candidate GPU parity probes must pass')
    fleet=inventory()
    if fleet['own_gpu_count']>3:raise RuntimeError('Current account cap exceeded')
    affinity={}
    for pid in eligible[-1]['gpu_pids']:
        values=sorted(os.sched_getaffinity(pid))
        if len(values)>16:raise RuntimeError('CPU affinity exceeds16')
        affinity[str(pid)]=values
    receipt=dict(passed=True,scope='first_sustained_resource_and_original_prefix_acceptance_not_full100completion',
        live=live,post280_samples_checked=len(eligible),all_post280_limits_passed=True,
        startup_samples_included=True,original_bounded_guard_still_active=True,candidate_traces=checks,
        candidate_gpu_parity=parity,cpu_affinity=affinity,current_inventory=fleet,
        old_zero_and_old_shared_not_rerun=True,independent_final_test_claimed=False)
    out=args.output or P/'target_candidate_eval_sustained_acceptance_20261025.json'
    if out.resolve().parent!=P.resolve() or out.exists():raise RuntimeError('Receipt must be a new file under reports')
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'seconds':live['seconds'],'post280_samples':len(eligible),'candidate_traces':checks,'receipt':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
