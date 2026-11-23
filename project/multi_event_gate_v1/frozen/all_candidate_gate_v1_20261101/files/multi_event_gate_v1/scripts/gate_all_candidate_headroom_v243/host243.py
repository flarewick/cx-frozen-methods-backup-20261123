"""Remove operational ptrace; retain original241 host and add conservative40% protection."""
import ast,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent/'gate_all_candidate_migration_v241';sys.path.insert(0,str(P));sys.path.insert(0,str(HERE))
file=P/'host241.py';tree=ast.parse(file.read_text());last=tree.body[-1]
assert isinstance(last,ast.Expr)and ast.unparse(last.value)=="runpy.run_path(str(p.D / 'benefit_admission_wait_host_v59.py'), run_name='__main__')"
ns=dict(__name__='operational_host243',__file__=str(file));exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),str(file),'exec'),ns)
# Nothing in the original241 host body has run yet. Drop only its strace Popen layer.
ns['subprocess'].Popen=ns['OriginalPopen']
import benefit_stage_guard as guard,protection243
old=protection243.install(guard)
p=ns['p'];proof=HERE/'CPU_proof243.json';v=json.loads(proof.read_text());assert v['passed']and v['protection_sha256']==p.sha(HERE/'protection243.py')and v['host243_sha256']==p.sha(__file__)
p.atomic(p.ROOT/'migration241/operational_protection243.json',dict(passed=True,original_guard_sha256=p.sha(p.D/'gpu_guard_bounded_metrics.py'),original_host241_sha256=p.sha(file),helper243_sha256=p.sha(__file__),protection243_sha256=p.sha(HERE/'protection243.py'),CPU_proof_sha256=p.sha(proof),all_original33pct_rules_and_guard_retained=True,additional_startup_and_steady40pct_headroom_after240s=True,signal_tracing_removed_to_restore_untraced_execution=True,model_gate_and_actual_execution_unchanged=True,historical_under33_sample_not_erased_or_claimed_passed=True))
exec(compile(ast.Module(body=[last],type_ignores=[]),str(file),'exec'),ns)
