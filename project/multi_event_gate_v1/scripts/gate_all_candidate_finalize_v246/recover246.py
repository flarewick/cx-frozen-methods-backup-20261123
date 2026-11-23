"""CPU-only recovery of an already completed, released GPU phase. Never restarts GPU."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;W=HERE.parents[1];R=W.parent;D=R/'deployment';H=W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4';cohort=int(os.environ.get('CX_GATE_TRANSFER_COHORT','0'));Q=H/('candidate_veto_all_events_v237_'+str(cohort))/'dev0'
sys.path.insert(0,str(D))
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def locate(f):return Path(str(f).replace('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA',str(R)))
def canonical(f):return '/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/'+str(Path(f).relative_to(R))
def atomic(f,v):t=f.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(f)
s=read(Q/'host_state.json');assert s['phase']=='stopped'and s['error']in ('Additional243 stop on unresolved PID metrics after240s','Additional243 GPU headroom below40% after240s; stop before risking original33% floor','GPU occupied before our verified arrival or arrival order unverified; stopping only our job')
b=Path(s['barrier']);t=Path(s['telemetry']);rows=[json.loads(x)for x in t.read_text().splitlines()];release=rows[-1];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
pids={pid for r in rows for pid in r.get('gpu_pids',[])};assert pids and all(not psutil.pid_exists(pid)for pid in pids)
import benefit_inventory_nvml_v60 as inv;inv.install();fleet=inv.inventory();assert all(not(pids&set(c['pids']))for c in fleet['cards'])
intent=b/'cpu_finalize_intent.json';v=read(intent);ready=read(b/'ready.json')
assert v['gpu_uuid']=='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a'and v['helper_sha256']==sha(D/'benefit_growth_deferred_finalize.py')and v['frozen_weights_verified_by_original_worker']and v['all_original_scene_threads_finished']
assert v['source_sha256']==ready['source_sha256']==v['plan']['source_sha256']
assert read(b/'mixed_expert_checkpoint.json')['all_expert_threads_finished']
for f,h in v['source_runtime_audits_sha256'].items():assert sha(locate(f))==h
whole=0
for j in v['plan']['jobs']:
 r=locate(j['run']);q=read(r/'concurrency_queue.json');n=len(q['completed']);assert n==len(read(r/'journal.json')['episodes'])and 0<=n<=20;whole+=n
assert whole>0
if whole<40:assert(b/'memory_recycle.json').exists()
for r in rows:
 if r.get('gpu_pids')and r.get('seconds',0)>=280:
  assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'],*r['process_mean_utilization'].values(),*r['process_rolling_300s_utilization'].values())>=33
  stamps=[x['unix_ns']/1e9 for x in r.get('process_metric_samples',[])if 'unix_ns'in x]
  if stamps and max(stamps)<v['created_unix']:assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'],*r['process_mean_utilization'].values(),*r['process_rolling_300s_utilization'].values())>=40
proof=dict(passed=True,CPU_only_recovery=True,no_GPU_restart=True,original_guard_stop_preserved=s['error'],all_GPU_PIDs_confirmed_exited=sorted(pids),actual_release=release,telemetry_sha256=sha(t),intent_sha256=sha(intent),actual_whole_G_committed=whole,complete40=(whole==40),normal_memory_drain=(whole<40),source_runtime_audits_verified=True,helper246_sha256=sha(__file__))
atomic(b/'post_release_recovery246.json',proof)
rr=b/'host_gpu_release_receipt.json';assert not rr.exists()or read(rr).get('operational_post_GPU_release_recovery246')
atomic(rr,dict(GPU_job_released=True,release=release,telemetry=canonical(t),telemetry_sha256=sha(t),intent_sha256=sha(intent),host_sha256=sha(D/'benefit_growth_host_v3.py'),original_host_sha256_field_is_compatibility_binding=True,operational_post_GPU_release_recovery246=True,original_guard_exit_success=False,recovery_proof_sha256=sha(b/'post_release_recovery246.json')))
log=b/'CPU_recovery246.log'
with log.open('w')as out:result=subprocess.run([sys.executable,'-u','-B',str(HERE.parent/'gate_all_candidate_migration_v241/cpu241.py'),str(HERE/'cpu246.py'),canonical(intent)],env=dict(os.environ,CX_GATE_TRANSFER_COHORT=str(cohort)),stdout=out,stderr=subprocess.STDOUT)
print(log.read_text()[-3000:],flush=True);assert result.returncode==0,'Original CPU finalizer failed'
receipt=read(b/'cpu_finalize_receipt.json');assert receipt['passed']and receipt['cuda_initialized']is False and receipt['intent_sha256']==sha(intent)
proof.update(original_CPU_finalizer_passed=True,CPU_finalize_receipt_sha256=sha(b/'cpu_finalize_receipt.json'));atomic(b/'post_release_recovery246.json',proof);r=read(rr);r['recovery_proof_sha256']=sha(b/'post_release_recovery246.json');atomic(rr,r)
s.update(phase='finished_pending_scientific_audit'if whole==40 else'maintenance_batch_complete',normal_drained_exit=whole<40,GPU_job_released=True,CPU_finalize_receipt=str(b/'cpu_finalize_receipt.json'),operational_recovery246=str(b/'post_release_recovery246.json'),historical_guard_error=s['error'],error=None,original_guard_exit_success=False);atomic(Q/'host_state.json',s)
print('246_ORIGINAL_CPU_FINALIZATION_PASSED',flush=True)
