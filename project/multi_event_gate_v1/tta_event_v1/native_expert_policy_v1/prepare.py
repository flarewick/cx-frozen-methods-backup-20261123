"""Freeze actual-state model inputs separately from offline expert labels."""
from native_expert_policy_v1.paths import *

def prepare():
    manifestfile=T/'outputs/native_expert_queries_v1/manifest.json'
    manifest=read(manifestfile)
    assert manifest['complete'] and manifest['counts']['queries']==36
    assert manifest['last_observation_after_success_not_a_policy_query']
    replayfile=T/'outputs/native_expert_smoke_v2/protocol.json'
    replay=read(replayfile)
    oldfile=T/'outputs/native_feedback_confirm_v5/protocol.json';old=read(oldfile)
    jobs=[];inputs={};sources=dict(old['source_sha256'])
    for scene in replay['scenes']:
        rows=[r for r in manifest['records'] if r['case_id']==scene['case_id']]
        assert rows and [r['query'] for r in rows]==list(range(len(rows)))
        assert all(r['split']=='adapter_fit' for r in rows)
        queries=[]
        for r in rows:
            f=r['observation_file'];assert sha(f)==r['observation_sha256'];inputs[f]=sha(f)
            queries.append(dict(query=r['query'],tick=r['observation_global_tick'],observation_file=f,
                                observation_sha256=r['observation_sha256']))
        jobs.append(dict(case_id=scene['case_id'],instruction=scene['instruction'],queries=queries))
    assert len(jobs)==4 and sum(len(j['queries']) for j in jobs)==36
    placementfile=T/'outputs/native_feedback_confirm_v5/cpu_placement_plan.json'
    placement=read(placementfile)
    for f in (manifestfile,replayfile,oldfile,placementfile,T/'native_resource_policy_v1/policy.json',
              T/'native_resource_policy_v1/guard.py',T/'native_terminal_guard_v1/strict_guard.py',
              Path(old['projection']),Path(old['training'])/'adapter_seed0.pt',
              P/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log'):
        inputs[str(f)]=sha(f)
    for f in C.rglob('*'):
        if f.is_file() and f.suffix in ('.py','.sh'):sources[str(f.relative_to(P))]=sha(f)
    for f in [T/'native_nextchunk_v1'/n for n in ('sampler_fork.py','query_ledger.py','query_current.py','child_exit.py')]+[P/'deployment'/n for n in ('adapter_cache_cpu.py','startup_loading.py','concurrent_rng.py')]:
        sources[str(f.relative_to(P))]=sha(f)
    spec=dict(version='native_expert_policy_v1',jobs=jobs,queries=36,seed=17,
        projection=old['projection'],training=old['training'],model_identity=old['model_identity'],
        required_gpu_uuid=placement['gpu_uuid'],CPU_placement=str(placementfile),
        source_sha256=sources,input_sha256=inputs,query_manifest=str(manifestfile),
        base_sampler='one actual infer(noise=None) per recorded real expert pre-execution state',
        sampler_seed='fresh17 per original case, sequential own query stream; no archived RNG import',
        encoder='frozen original batch3; Observer current-frame/current-state anchor, current joint[0] to head',
        gpu_deadline_seconds=240,no_simulator=True,no_future_labels_in_worker_inputs=True,
        scope='Four fixed originalFIT scenes only. Offline base comparator and actual hidden64 on36 previously visited states. Sampled base was NOT executed and did NOT generate later expert observations. No claimed online query or causal feedback validation.',
        native_training_cache_ready=False,training_steps=0,online_updates=0,TTA_utility_proven=False)
    spec['identity']=identity(spec);f=O/'protocol.json'
    if f.exists():assert read(f)==spec,'Frozen expert policy protocol changed'
    else:write(f,spec)
    verify(spec);return spec

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    s=prepare();root,m=dataset(s,a.gpu_uuid)
    print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 36)),flush=True)
