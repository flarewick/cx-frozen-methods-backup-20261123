"""Reversible, enumerated guard edits; every other original AST node exact."""
import ast,copy

def transform(source):
 original=ast.parse(source);tree=copy.deepcopy(original)
 imports=[n for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)and n.module=='native_fast_inventory_v1.inventory'];assert len(imports)==1;imports[0].module='native_account_stream_v1.transport'
 target="cumulative, rolling = utilization_window.add(elapsed, max(sm, bandwidth), {pid: pid_util.get(pid, 0) for pid in active})"
 hits=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)and ast.unparse(n)==target];assert len(hits)==1
 before=ast.parse("missing_zero = admissible_missing(active,pid_util,metric_samples,info['uuid'],gpu)\nzeroed_interval = zero_previous(utilization_window,recent_window,missing_zero)").body
 telemetry=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='telemetry'for t in n.targets));assert isinstance(telemetry.value,ast.Dict)
 extras={'missing_process_zero_lower_bound':ast.Name(id='missing_zero',ctx=ast.Load()),'zeroed_missing_interval':ast.Name(id='zeroed_interval',ctx=ast.Load()),'lower_bound_exact_elapsed':ast.Name(id='elapsed',ctx=ast.Load())}
 for k,v in extras.items():telemetry.value.keys.append(ast.Constant(k));telemetry.value.values.append(v)
 checks=[n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=='elapsed >= 280 and active'];assert len(checks)==1
 predicates=[n for n in ast.walk(checks[0])if isinstance(n,ast.Compare)and ast.unparse(n)=='pid not in pid_util'];assert len(predicates)==1
 replacement=ast.parse('pid not in pid_util and pid not in missing_zero',mode='eval').body
 class Edit(ast.NodeTransformer):
  def visit_Assign(self,n):
   if n is hits[0]:return before+[n]
   return self.generic_visit(n)
  def visit_Compare(self,n):
   if n is predicates[0]:return copy.deepcopy(replacement)
   return self.generic_visit(n)
 changed=Edit().visit(tree);restored=copy.deepcopy(changed)
 class Undo(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='native_account_stream_v1.transport':n.module='native_fast_inventory_v1.inventory'
   return n
  def visit_Assign(self,n):
   if len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('missing_zero','zeroed_interval'):return None
   if len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='telemetry':
    kept=[(k,v)for k,v in zip(n.value.keys,n.value.values)if not(isinstance(k,ast.Constant)and k.value in extras)];n.value.keys=[k for k,v in kept];n.value.values=[v for k,v in kept]
   return self.generic_visit(n)
  def visit_BoolOp(self,n):
   if ast.dump(n)==ast.dump(replacement):return ast.parse('pid not in pid_util',mode='eval').body
   return self.generic_visit(n)
 assert ast.dump(Undo().visit(restored))==ast.dump(original),'Unexpected guard mutation'
 return ast.fix_missing_locations(changed)
