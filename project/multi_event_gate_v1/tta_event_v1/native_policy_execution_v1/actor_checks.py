"""Persistent actor ledger wiring on a real prior command receipt, no new infer."""
from pathlib import Path
import copy,json,numpy as np
from .actor import ActorQuery
from .common import T,read,sha,write
from native_nextchunk_v1.query_ledger import QueryLedger

def main():
 index=read(T/'outputs/native_expert_full_feedback_v1/index.json');case=next(c for c in index['cases']if not c['old_pilot_reused']and c['native_available']);row=case['queries'][0];witness=row['policy']['noise_witness'];trace=row['trace']
 owner=ActorQuery.__new__(ActorQuery);owner.rows=[row['policy']];owner.ledger=QueryLedger('CPU_receipt_binding_fixture');owner.poisoned=False
 owner.ledger.record_draw(0,0,witness['base_sha256'],witness['rng_before'],witness['rng_after'],actual_sampler_call=True)
 try:owner.ledger.permit_query(1,trace['committed_ticks'])
 except ValueError:pass
 else:raise AssertionError('Draw admitted before execution')
 with np.load(trace['file'],allow_pickle=False)as z:commands=z['source_commands'].copy()
 assert len(commands)==32 and trace['complete_source_chunk']
 bad=commands.copy();bad[0,0]+=1
 out=T/'outputs/native_policy_execution_v1/actor_receipt_fixture';assert not out.exists()
 try:owner.finish_chunk(bad,trace,trace['file'],out)
 except ValueError:pass
 else:raise AssertionError('Changed command plan accepted')
 assert owner.ledger.awaiting_execution
 proof=owner.finish_chunk(commands,trace,trace['file'],out)
 assert proof['actual_complete_native32']and proof['commands_returned']==32 and not owner.ledger.awaiting_execution
 assert owner.ledger.permit_query(1,proof['end_tick'])
 try:owner.finish_chunk(commands,trace,trace['file'],out)
 except ValueError:pass
 else:raise AssertionError('Same execution committed twice')
 write(T/'outputs/native_policy_execution_v1/CPU_actor_receipt_checks.json',dict(passed=True,actual_prior_expert_receipt_binding_fixture=True,not_an_actor_rollout=True,changed_command_rejected=True,query_before_execution_rejected=True,duplicate_execution_rejected=True,new_VLA_draws=0,new_native_rollouts=0,source_sha256={str(p):sha(p)for p in [Path(__file__),Path(__file__).with_name('actor.py'),Path(__file__).with_name('common.py')]}))
 print('PERSISTENT_ACTOR_QUERY_RECEIPT_BINDING_PASS',flush=True)
if __name__=='__main__':main()
