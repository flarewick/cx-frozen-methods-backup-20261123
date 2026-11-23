"""Read the detached optimized-laptop comparison without touching any GPU."""
import json
from pathlib import Path
from evaluate import host
D=Path(__file__).resolve().parent;P=D/'reports'

def status():
    path=P/'target_candidate_eval_active_20261025.json'
    if not path.exists():return dict(phase='not_launched',gpu_started=False)
    active=json.loads(path.read_text());proc=Path('/proc')/str(active['host_pid'])/'stat';alive=False
    if proc.exists():
        values=proc.read_text().split();alive=values[2]!='Z' and values[21]==str(active['host_start_ticks'])
    file=Path(active['status_file']);state=json.loads(file.read_text()) if file.exists() else dict(phase='launching')
    counts={}
    for job in state.get('jobs',[]):
        root=host(job['run']);q=root/'concurrency_queue.json';j=root/'journal.json';episodes=[]
        if q.exists():episodes=list(json.loads(q.read_text())['completed'].values())
        elif j.exists():episodes=json.loads(j.read_text())['episodes']
        counts[job['condition']]=dict(completed=len(episodes),requested=100,successes=sum(e['success'] for e in episodes),run=str(root))
    resources=state.get('resources') or {}
    return dict(phase=state['phase'],master_running=alive,host_pid=active['host_pid'],gpu=active['gpu'],conditions=counts,
        seconds=resources.get('seconds'),process_mean_utilization=resources.get('process_mean_utilization'),process_rolling_300s_utilization=resources.get('process_rolling_300s_utilization'),
        device_cumulative_utilization=resources.get('time_weighted_mean_utilization'),recent_60s_utilization=resources.get('process_recent_60s_utilization'),memory_mib=resources.get('memory_mib'),cpu=resources.get('cpu'),rss_bytes=resources.get('rss_bytes'),account_gpu_count=resources.get('account_gpu_count'),
        telemetry=state.get('telemetry'),log=active['log'],error=state.get('error'),historical_guard_error=state.get('historical_guard_error'))

if __name__=='__main__':print(json.dumps(status(),ensure_ascii=False,indent=2))
