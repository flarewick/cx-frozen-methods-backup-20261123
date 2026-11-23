"""Freeze actual expert origin/following frame capture without new model draws."""
from native_expert_frames_v1.paths import *
def prepare():
 original=T/'outputs/native_expert_smoke_v2';old=read(original/'protocol.json')
 roots=[f.parent for f in(original/'collections').glob('*/manifest.json')if read(f)['protocol_identity']==old['identity']];assert len(roots)==1;r=roots[0]
 assert read(r/'result.json')['complete']and read(r/'interface_qualification.json')['passed']
 framefile=r/'GPU_frames_complete.json';frames=read(framefile);phasefile=T/'outputs/native_expert_clock_audit_v1/report.json';phases=read(phasefile)
 assert phases['complete']and phases['counts']['queries']==36
 terminal=T/'outputs/native_expert_policy_v1/terminal_audit.json';assert read(terminal)['passed']
 qfile=T/'outputs/native_expert_queries_v1/manifest.json';query=read(qfile)
 pfile=T/'outputs/native_expert_policy_v1/protocol.json';policy=read(pfile)
 sourceaudit=T/'outputs/native_expert_policy_v1/closing_source_audit.json';assert read(sourceaudit)['passed']
 sources=dict(old['source_sha256']);inputs=dict(old['input_sha256']);scenes=[]
 for scene,source in zip(old['scenes'],frames['rows'],strict=True):
  assert scene['case_id']==source['case_id']and source['available'];n=len(source['chunks'])
  events=[p for p in phases['rows']if p['case_id']==scene['case_id']and p['event']is not None]
  assert len(events)==4 and all(len(p['event']['kinds'])==1 for p in events)
  kinds={str(p['query']):p['event']['kinds'][0]for p in events};dense=sorted({int(q)for q in kinds}|{int(q)+1 for q in kinds if int(q)+1<n})
  scenes.append(dict(scene,reference_capture=source,event_kinds=kinds,dense_queries=dense,
                     query_references=[q for q in query['records']if q['case_id']==scene['case_id']]))
  for obs in source['observations']:assert sha(obs['file'])==obs['sha256'];inputs[obs['file']]=obs['sha256']
  for chunk in source['chunks']:assert sha(chunk['file'])==chunk['sha256'];inputs[chunk['file']]=chunk['sha256']
 assert sum(len(s['dense_queries'])for s in scenes)==28
 for f in C.rglob('*'):
  if f.is_file()and f.suffix in('.py','.sh'):sources[str(f.relative_to(P))]=sha(f)
 for f in(T/'native_expert_smoke_v2/setup.py',T/'native_expert_smoke_v2/observe.py',T/'native_feedback_confirm_v5/cx_tta_feedback_confirm5/capture.py',T/'stage_feedback_v3/completion.py',T/'native_stage_v1/clock.py'):
  sources[str(f.relative_to(P))]=sha(f)
 for f in(framefile,phasefile,terminal,qfile,pfile,sourceaudit,r/'result.json',r/'interface_qualification.json'):inputs[str(f)]=sha(f)
 spec=dict(version='native_expert_frames_v1',scenes=scenes,source_sha256=sources,input_sha256=inputs,
  runtime_source=old['runtime_source'],required_gpu_uuid=old['required_gpu_uuid'],CPU_placement=old['CPU_placement'],
  max_ticks_per_chunk=5000,gpu_deadline_seconds=7200,queries=36,dense_chunks=28,
  real_readouts='Frozen COMMON_TICKS plus actual requested95 plus actual terminal; headRGB only between queries',
  exact_existing_query_and_trace_binding_required=True,actual_expert_commands_define_requested_phase=True,
  no_VLA_or_semantic_model_loaded=True,no_new_model_draws=True,no_extra_actions_or_holds=True,
  scope='Four originalFIT teacher executions with real missing arrived RGB. Offline later causal tracking allowed only after prefix/same-query proof. No claim current online adaptation or efficacy.',
  native_training_cache_ready=False,training_steps=0,online_updates=0,TTA_utility_proven=False)
 spec['identity']=identity(spec);f=O/'protocol.json'
 if f.exists():assert read(f)==spec,'Frozen frame protocol changed'
 else:write(f,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 4)),flush=True)
