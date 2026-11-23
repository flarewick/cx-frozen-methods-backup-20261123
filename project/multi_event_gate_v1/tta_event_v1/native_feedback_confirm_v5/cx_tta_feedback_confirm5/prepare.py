"""Fixed sources/recipe before new exposure; execution files bound after CPU preparation."""
from pathlib import Path
import argparse,copy,json
import numpy as np
from .paths import CODE,TTA,PROJECT,OUTPUT,read,write,digest,identity,verify
from .contract import dataset,load_journal
from native_cpu_seed_v4.store import PreparedCPUSeedStore
CANDIDATE=TTA/'outputs/native_feedback_candidate_v1/frozen_candidate.json'
def locate_initial(candidate):
 base=TTA/'outputs/native_initial_confirm_v1';spec=read(base/'protocol.json')
 assert spec['candidate_identity']==candidate['identity']
 roots=[p.parent for p in(base/'collections').glob('*/manifest.json')if read(p)['protocol_identity']==spec['identity']]
 assert len(roots)==1;root=roots[0]
 assert read(root/'result.json')['complete']and read(root/'interface_qualification.json')['passed']
 raw=read(root/'GPU_frames_complete.json');assert raw['complete']and len(raw['rows'])==32
 return spec,root,raw
def prepare(cpu_fixture=False):
 oldfile=TTA/'outputs/native_online_bridge_v3/protocol.json';old=read(oldfile)
 inputs=dict(old['input_sha256']);inputs.update(old['source_sha256']);inputs[str(oldfile)]=digest(oldfile)
 bridge_root=TTA/'outputs/native_initial_frame_v3/collections/e56555e2e8724ece6a62a1bf'
 assert read(bridge_root/'result.json')['complete']and read(bridge_root/'interface_qualification.json')['passed']
 bridge_rows={r['case_id']:r for r in read(bridge_root/'GPU_frames_complete.json')['rows']}
 bridge_cache=TTA/'outputs/native_cpu_seed_v4/developer_bridge/prepared'
 assert read(bridge_cache.parent/'audit.json')['passed'];oldstore=PreparedCPUSeedStore(bridge_cache)
 scenes=[]
 for item in old['scenes']:
  item=copy.deepcopy(item);row=bridge_rows[item['case_id']]
  with np.load(row['file'],allow_pickle=False)as z:images=[z['cam'+str(i)].copy()for i in range(3)]
  seed,_=oldstore.at_initial_observation(images,item['instruction'],query=0,tick=0);assert seed['available']
  # Keep only deterministic seed data, not new timestamp/lookup duration in protocol identity.
  for key in('accepted_monotonic_ns','lookup_seconds'):seed.pop(key)
  item.update(earliest_seed=seed,role='excluded_CPU_initialization_and_runtime_bridge',
   initial_observation_file=row['file'],initial_observation_sha256=row['sha256'])
  scenes.append(item)
  for p in(row['file'],seed['mask_file']):inputs[str(p)]=digest(p)
 candidate=None;initial_missing=[];initial_cases=[]
 if not cpu_fixture:
  if not CANDIDATE.exists():raise RuntimeError('Complete scientific candidate freeze required before independent policy collection')
  candidate=read(CANDIDATE);assert candidate['frozen']and candidate['full_feedback_candidate_frozen']
  for p,h in candidate['source_sha256'].items():assert digest(p)==h
  init_spec,init_root,raw=locate_initial(candidate);inputs[str(CANDIDATE)]=digest(CANDIDATE)
  prepared=TTA/'outputs/native_cpu_seed_v4/independent_confirmation/prepared';store=PreparedCPUSeedStore(prepared)
  assert store.result['requested']==32 and store.protocol['scope']=='independent_TRAIN_confirmation'
  manifest_file=prepared.parent/'manifest.json';assert digest(manifest_file)==store.protocol['manifest_sha256']
  assert read(manifest_file)['candidate_identity']==candidate['identity']
  bycase={r['case_id']:r for r in raw['rows']};initial_cases=init_spec['scenes']
  for item in init_spec['scenes']:
   row=bycase[item['case_id']]
   if not row['available']:
    initial_missing.append(dict(case_id=item['case_id'],reason=row['reason']));continue
   with np.load(row['file'],allow_pickle=False)as z:images=[z['cam'+str(i)].copy()for i in range(3)]
   seed,_=store.at_initial_observation(images,item['instruction'],query=0,tick=0)
   for key in('accepted_monotonic_ns','lookup_seconds'):seed.pop(key)
   scenes.append(dict(item,role='independent_TRAIN_confirmation',source_family='current_RGB_native',legacy_full32=False,
    requires_sequential_reference=False,teacher_records={},old_stage_teacher_available=False,
    initial_physical_sha256=row['initial_physical_sha256'],initial_observation_file=row['file'],
    initial_observation_sha256=row['sha256'],earliest_seed=seed))
   inputs[str(row['file'])]=row['sha256']
   if seed['available']:inputs[seed['mask_file']]=seed['mask_sha256']
  assert all(sum(x['task']==task for x in scenes[2:])>=8 for task in('stack_blocks_two','stack_bowls_two')),'Insufficient raw initialization coverage; retain missing cases, no replacements'
  # Scheduling only: four predefined longest prefixes keep useful work available during bridge validation.
  ordered=sorted(scenes[2:],key=lambda r:(-r['coverage_start_query'],r['seed'],r['task']))
  scenes=scenes[:2]+ordered
  for p in(init_root/'GPU_frames_complete.json',init_root/'interface_qualification.json',prepared/'protocol.json',prepared/'result.json'):
   inputs[str(p)]=digest(p)
 for p in(bridge_root/'result.json',bridge_root/'GPU_frames_complete.json',bridge_cache/'protocol.json',bridge_cache/'result.json',bridge_cache.parent/'audit.json',
  OUTPUT/'cpu_placement_plan.json',CODE/'COPIED_SOURCE.json'):
  inputs[str(p)]=digest(p)
 if not cpu_fixture:
  recovery_file=TTA/'outputs/native_feedback_recovery_v2/whole_case_audit.json';recovery=read(recovery_file)
  assert recovery['complete']and recovery['passed']and recovery['whole_cases_verified']==20
  assert recovery['original_actual_host_exit']==1 and not recovery['original_resource_qualification_passed']
  assert recovery['old_failure_and_source_files_unchanged']and recovery['no_partial_branch_splicing']
  assert recovery['complete_requests_selected_exhaustively']and len(recovery['remaining'])==12
  originalfile=TTA/'outputs/native_feedback_confirm_v4/protocol.json';original=read(originalfile)
  assert scenes==original['scenes'],'Scheduling adaptation must not change any case input'
  assert [r['index']for r in recovery['rows']]==list(range(17))+[18,20,21]
  indices=[r['index']for r in recovery['rows']]+[r['index']for r in recovery['remaining']]
  assert sorted(indices)==list(range(len(scenes)))
  scenes=[scenes[i]for i in indices]
  assert [r['case_id']for r in recovery['rows']]==[r['case_id']for r in scenes[:20]]
  inputs[str(recovery_file)]=digest(recovery_file);inputs[str(originalfile)]=digest(originalfile)
 for p in (TTA/'native_resource_policy_v1/policy.json',TTA/'native_resource_policy_v1/guard.py',TTA/'native_terminal_guard_v1/strict_guard.py'):
  inputs[str(p)]=digest(p)
 sources={str(p.relative_to(PROJECT)):digest(p)for p in CODE.rglob('*')if p.is_file()and p.suffix in('.py','.sh')}
 for family in('native_cpu_seed_v4','native_shared_response_v1'):
  for p in(TTA/family).glob('*.py'):sources[str(p.relative_to(PROJECT))]=digest(p)
 drop={'identity','source_sha256','input_sha256','scenes','version','purpose','runtime_status','acceptance_gate'}
 spec={k:v for k,v in old.items()if k not in drop}
 spec.update(version='native_feedback_confirm_v5',cpu_fixture=bool(cpu_fixture),scenes=scenes,bridge_count=2,simulation_lanes=4,
  source_sha256=sources,input_sha256=inputs,CPU_admission_wait_seconds=7200,required_gpu_uuid=read(OUTPUT/'cpu_placement_plan.json')['gpu_uuid'],
  candidate_identity=None if cpu_fixture else candidate['identity'],original_requested_independent_cases=0 if cpu_fixture else 32,
  initial_cases=initial_cases,initial_missing_cases=initial_missing,
  purpose='Two excluded old full3chunk bridges + fixed new32 TRAIN feedback confirmation. Full scientific candidate frozen before any new q0. No online optimizer or trained-response application.',
  acceptance_gate='Old bridges exact all physical/RNG/current actor/three chunks and sequential reference; all-case own q0 raw binding, Z/sham, CPU feedback audit, actual normal child/release/host0 and strict resource qualification; independent physical-feedback metric gate afterward.',
  runtime_status='CPU_PREFLIGHT_OLD_METADATA_ONLY'if cpu_fixture else'FROZEN_INDEPENDENT_TRAIN_CONFIRMATION',
  full_online_candidate_frozen=False,scientific_feedback_candidate_frozen=not cpu_fixture,
  new_independent_scenes=0 if cpu_fixture else 32,direction_training=False,online_updates=0,TTA_utility_proven=False)
 spec['identity']=identity(spec);p=OUTPUT/('preflight_protocol.json'if cpu_fixture else'protocol.json')
 if p.exists():assert read(p)==spec,'Frozen confirmation execution protocol changed'
 else:write(p,spec)
 verify(spec);print('FIXED_FEEDBACK_CONFIRMATION_PROTOCOL',spec['identity'],len(scenes),bool(cpu_fixture),flush=True);return spec
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid');p.add_argument('--cpu-fixture',action='store_true');a=p.parse_args();spec=prepare(a.cpu_fixture)
 if a.gpu_uuid:
  root,m=dataset(spec,a.gpu_uuid);j=load_journal(root,m,spec)
  print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],completed=len(j['cases']),remaining=len(spec['scenes'])-len(j['cases']))),flush=True)
