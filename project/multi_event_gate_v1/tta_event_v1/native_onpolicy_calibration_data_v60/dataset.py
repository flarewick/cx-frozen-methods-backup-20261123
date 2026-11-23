from native_fit_support_review_v1.support import minimum_fit,evidence as support_evidence
"""Original fixed population retained, prospective new75 support declared in advance."""
from pathlib import Path
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .reader import Reader
TASKS=('beat_block_hammer','turn_switch','adjust_bottle','grab_roller','open_laptop')
BASE=tuple(p.name for p in sorted((T/'outputs').glob('native_onpolicy_collection_v[0-9]*'),key=lambda p:int(p.name.rsplit('_v',1)[1])if p.name.rsplit('_v',1)[1].isdigit()else -1,reverse=True)if p.is_dir()and p.name.rsplit('_v',1)[1].isdigit())

ARM_BASE=tuple(p.name for p in sorted((T/'outputs').glob('native_onpolicy_arm_collection_v[0-9]*'),key=lambda p:int(p.name.rsplit('_v',1)[1])if p.name.rsplit('_v',1)[1].isdigit()else -1,reverse=True)if p.is_dir()and p.name.rsplit('_v',1)[1].isdigit())

def complete_data(*,include_arrays=False):
 reader=Reader();records=[];bindings={};roots=[]
 for population in('original_fixed250','initial_ARM_fixed75'):
  for split in('fit','val'):
   original=population=='original_fixed250';start=(7120000 if split=='fit'else 7121000)if original else(7124000 if split=='fit'else 7125000);count=(40 if split=='fit'else 10)if original else(10 if split=='fit'else 5);stages=(8 if split=='fit'else 2)if original else(2 if split=='fit'else 1)
   candidates=[T/'outputs'/version/(split+'_complete.json')for version in BASE]if original else[T/'outputs'/version/(split+'_complete.json')for version in ARM_BASE]
   done=next((p for p in candidates if p.exists()),None)
   if done is None:raise ValueError('Full fixed325 original250 plus initial-ARM75 absent; fitting forbidden: '+population+'/'+split)
   report=read(done);assert report['complete']and report['cases_each_task']==count and len(report['stages'])==stages;bindings[str(done)]=sha(done)
   for stage in report['stages']:
    root=mapped(stage['root']);actual=mapped(stage['actual_host_exit_file']);assert sha(actual)==stage['actual_host_exit_sha256'];spec,cases,audit=reader.qualified(root)
    assert spec['actual_seed_start']==start and spec['dataset_split']==('adapter_fit'if split=='fit'else'adapter_validation')
    assert bool(spec.get('initial_ARM_path_feedback'))==(not original)
    roots.append(dict(root=str(root),split=split,population=population,protocol_identity=spec['identity']));bycase={r['case_id']:r for r in audit['records']}
    for case in cases:
     file=root/'accepted'/(case['case_id']+'.json');accepted=read(file);bindings[str(file)]=sha(file);row=bycase[case['case_id']];examples=[]
     if row['label_available']:
      value=reader.example(root,spec,case,row);p=row['packet'];key=(case['case_id'],p['source_query'],p['query'],row['query']);e=dict(key=key,application_mode=value['application_mode'],feedback_query=value['actual_feedback_query'],action_query=value['offline_action_query'])
      if include_arrays:e['arrays']=value
      examples.append(e)
     records.append(dict(case_id=case['case_id'],task=accepted['task'],seed=accepted['seed'],index=accepted['episode_index'],split=spec['dataset_split'],population=population,examples=examples,no_policy_outcome_filter=True,teacher_kind=row.get('teacher_kind','reference_suffix'),missing_reason=None if examples else row['reason']))
   for task in TASKS:
    rr=[r for r in records if r['population']==population and r['task']==task and r['split']==('adapter_fit'if split=='fit'else'adapter_validation')]
    assert len(rr)==count and {r['index']for r in rr}==set(range(count))and all(start<=r['seed']<start+1000 for r in rr)
 assert len(records)==325 and len({r['case_id']for r in records})==325
 assert sum(r['population']=='original_fixed250'for r in records)==250 and sum(r['population']=='initial_ARM_fixed75'for r in records)==75
 groups={}
 for task in TASKS:
  groups[task]={}
  for split,count,minimum in(('adapter_fit',50,minimum_fit()),('adapter_validation',15,1)):
   rr=[r for r in records if r['task']==task and r['split']==split];assert len(rr)==count;covered={r['case_id']for r in rr if r['examples']};assert len(covered)>=minimum,('Missing distinct eligible actual full-state support',task,split,len(covered))
   groups[task][split]=dict(identities=count,eligible_distinct_cases=len(covered),packets=sum(len(r['examples'])for r in rr))
 keys=[tuple(e['key'])for r in records for e in r['examples']];assert len(keys)==len(set(keys));bindings.update(reader.inputs)
 bindings.update(support_evidence())
 bindings.update({str(p):sha(p)for p in (T/'native_audit_path_identity_v1').glob('*.py')})
 return records,dict(minimum_distinct_FIT_cases=minimum_fit(),exploratory_support_exception=minimum_fit()==7,complete=True,identities=325,FIT=250,VAL=75,groups=groups,stages=roots,original_fixed250_retained=True,predeclared_initial_ARM75=True,short_mechanism_probes_and_EVAL_excluded=True,no_policy_outcome_filter=True,input_sha256=bindings)

def main():
 records,value=complete_data();write(T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset_pending_actual_build_exit.json',dict(**value,records=records,optimizer_steps=0,actual_build_host_exit_pending=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}))
if __name__=='__main__':main()
