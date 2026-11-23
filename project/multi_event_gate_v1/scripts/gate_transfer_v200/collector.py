"""Original bounded collectors; change CPU audit runner only, scope edits recorded."""
import ast,copy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import entry;entry.install()
import benefit_followup_protocol_v2 as p
name='benefit_event_full_G_collect_v174.py'if sys.argv[1]=='validate'else'benefit_event_full_collect_v174.py';sys.argv=[str(p.D/name),*sys.argv[2:]]
# entry.transform verifies folder/deferred-path edits; the dispatcher below changes only final CPU runner.
code=entry.transform(p.D/name);ns=dict(__name__='transfer_collector',__file__=str(p.D/name));exec(code,ns)
# Rewrite just main's audit dispatch, preserving original callable globals and original retry/guards.
import inspect,linecache
original=ast.parse(inspect.getsource(ns['main']));tree=copy.deepcopy(original);hits=[]
for n in ast.walk(tree):
 if isinstance(n,ast.List)and any(isinstance(x,ast.Constant)and x.value=='benefit_event_full_scope_audit_v175.py'for x in n.elts):
  assert ast.unparse(n.elts[0])=="p.D / 'benefit_followup_cpu_host_v6.py'";before=copy.deepcopy(n.elts[:5]);n.elts=[ast.Constant(str(HERE/'cpu_host.py')), *n.elts[5:]];hits.append(before)
assert len(hits)==1
restored=copy.deepcopy(tree)
for n in ast.walk(restored):
 if isinstance(n,ast.List)and n.elts and isinstance(n.elts[0],ast.Constant)and n.elts[0].value==str(HERE/'cpu_host.py'):n.elts=copy.deepcopy(hits[0])+n.elts[1:]
assert ast.dump(restored)==ast.dump(original)
exec(compile(ast.fix_missing_locations(tree),'<transfer200_collector_CPU_dispatch>','exec'),ns);ns['main']()
