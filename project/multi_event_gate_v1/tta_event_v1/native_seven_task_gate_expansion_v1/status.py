"""Read-only monitor for new isolated campaigns; never starts GPU work."""
import argparse,collections,json,re
from .contract import T,read,mapped

def status(version,stage):
    assert re.fullmatch(r'native_seven_[a-z0-9_]+',version)and re.fullmatch(r'[a-z0-9_]+',stage)
    b=T/'outputs'/version/stage;d=dict(version=version,stage=stage)
    if(b/'study.json').exists():d['target_cases']=read(b/'study.json')['cases_per_batch']
    if(b/'live_status.json').exists():
        x=read(b/'live_status.json');d.update(phase=x.get('phase'),gpu=x.get('gpu'),resources={k:x.get('resources',{}).get(k)for k in('seconds','gpu_pids','memory_mib','account_gpu_count','process_rolling180')})
    if(b/'execution_state.json').exists():
        root=mapped(read(b/'execution_state.json')['root']);d['root']=str(root)
        d['finished']={c:len(list(root.glob('cases/*/'+c+'/result.json')))for c in('baseline','M_event_full','TTA','zero_update','shadow_update','handoff_reference')}
        f=root/'owner_profile_runtime.json';f=f if f.exists()else root/'owner_profile_progress.json'
        if f.exists():
            groups=collections.defaultdict(lambda:[0,0.])
            for row in read(f)['rows']:groups[row['function']][0]+=row['calls'];groups[row['function']][1]+=row['exclusive_host_ns']/1e9
            d['exclusive_host_NOT_GPU_kernel']=sorted([dict(function=k,calls=v[0],seconds=round(v[1],3))for k,v in groups.items()],key=lambda x:-x['seconds'])[:5]
    if(b/'actual_host_wait_exit.json').exists():d.update(terminal=True,actual=read(b/'actual_host_wait_exit.json'),report_exists=(b/'pilot_complete.json').exists())
    print(json.dumps(d,ensure_ascii=False),flush=True);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--stage',required=True);a=p.parse_args();status(a.version,a.stage)
