"""Only the instantaneous empty-idle predicate changes; all averages remain exact."""
import ast,copy

def transform_text(source):
 old="r['process_utilization'][k]is not None"
 assert source.count(old)==1
 return source.replace(old,"(r['process_utilization'][k]is not None or idle_sample_eligible(r,k))")
def terminal_ast_exact(new,old):
 tree=copy.deepcopy(new);replacement=ast.parse("r['process_utilization'][k]is not None or idle_sample_eligible(r,k)",mode='eval').body;hits=[]
 class Undo(ast.NodeTransformer):
  def visit_BoolOp(self,n):
   if ast.dump(n)==ast.dump(replacement):hits.append(n);return copy.deepcopy(replacement.values[0])
   return self.generic_visit(n)
 restored=Undo().visit(tree)
 assert len(hits)==1 and ast.dump(restored)==ast.dump(old),'Unexpected terminal predicate or threshold change'
 return True
