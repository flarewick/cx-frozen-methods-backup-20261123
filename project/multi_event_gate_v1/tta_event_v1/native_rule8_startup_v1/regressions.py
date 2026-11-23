"""Actual failed-row regression and conservative grace boundaries, no GPU."""
from pathlib import Path
from copy import deepcopy
import json
from .policy import evaluate,rule8_subject
from .checks import checks,check,stats
from native_governor180_v4.policy import evaluate as old_evaluate
from native_policy_execution_v1.common import write,sha

def main():
 T=Path(__file__).resolve().parents[1];C=Path(__file__).parent
 old=(T/'native_governor180_v4/policy.py').read_text();new=(C/'policy.py').read_text()
 start=new.index('def rule8_subject(');end=new.index('def evaluate(',start)
 restored=new[:start]+new[end:]
 restored=restored.replace("if rule8_subject(c, check_age) and not extra_ok(c, limits)","if not extra_ok(c, limits)")
 check('exact_policy_inverse_only_Rule8_creation_grace',restored==old)
 root=T/'outputs/native_seven_official_v6/plain_001/batches/000/runs/c3cdde17dccb4da6a79ec60e361cca0f'
 file=root/'attempt/governor180.jsonl';rows=[json.loads(s) for s in file.read_text().splitlines()]
 last=rows[-1];fleet=last['fleet'];owned={tuple(p) for p in last['owned_identities']}
 check('actual_original_Rule8_failure_preserved',old_evaluate(fleet,owned)==last['server_verdict'] and not last['server_verdict']['passed'])
 check('actual_second_GPU_age7_does_not_reject_old_GPU',evaluate(fleet,owned)['passed'])
 extra=next(c for c in fleet['cards'] if c['index']==2);p=extra['processes'][0]
 check('actual_second_creation_age_is_young',0<=p['age']<10 and p['pid']==3014133)
 for age in (279.999,280.,299.999,300.):
  f=deepcopy(fleet);q=next(c for c in f['cards'] if c['index']==2)['processes'][0];q['age']=age
  check('precheck_age_'+str(age),evaluate(f,owned)['passed']==(age<280))
  check('authoritative_age_'+str(age),evaluate(f,owned,precheck=False)['passed']==(age<300))
 for key,value in [('birth',None),('age',None),('age',-1),('age',float('nan'))]:
  c=deepcopy(extra);c['processes'][0][key]=value
  check('unknown_'+key+str(value)+'_not_exempt',rule8_subject(c,300))
 c=deepcopy(extra);c['processes']=[];check('missing_own_process_not_exempt',rule8_subject(c,300))
 c=deepcopy(extra);oldp=deepcopy(c['processes'][0]);oldp.update(pid=999999,birth=100,age=301);c['processes'].append(oldp);c['own_pids'].append(oldp['pid'])
 check('mixed_mature_and_young_card_still_subject',rule8_subject(c,300))
 f=deepcopy(fleet);c=next(c for c in f['cards'] if c['index']==2);c['processes'][0]['age']=300;c['own_rolling180']=stats(70,50)
 check('mature_other_GPU_exact70_AND50_allows_one_low_GPU',evaluate(f,owned)['passed'])
 c['own_rolling180']=stats(100,49.99);check('mature_high_util_low_VRAM_still_rejected',not evaluate(f,owned)['passed'])
 check('all_earlier_actual_verdicts_byte_exact',all(evaluate(v['fleet'],{tuple(p) for p in v['owned_identities']})==v['server_verdict'] for v in rows[:-1]))
 write(T/'outputs'/C.name/'CPU_checks_v1.json',dict(passed=True,no_GPU_work=True,old_failed_actual1_preserved=True,exact_policy_inverse=True,checks=checks,actual_raw_rows_replayed=len(rows),input_sha256={str(file):sha(file)},source_sha256={str(p):sha(p)for p in C.glob('*')if p.is_file()}))
 print('RULE8_CREATION_GRACE_CPU_PASS',len(checks),flush=True)
if __name__=='__main__':main()
