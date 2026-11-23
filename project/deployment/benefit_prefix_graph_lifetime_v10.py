"""Retain the actual captured constant owner; every original GPU operator unchanged."""
import ast,copy,hashlib,inspect,linecache,textwrap
_INSTALLED=False
RECEIPT=None

def install():
 global _INSTALLED,RECEIPT
 import benefit_prefix_graph as original
 if _INSTALLED:return RECEIPT
 if original._CACHE:raise RuntimeError('Lifetime repair must precede fresh graph capture')
 fn=original.hidden;node=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0];before=copy.deepcopy(node)
 targets=[n.value for n in ast.walk(node)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='entry'for t in n.targets)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='dict']
 if len(targets)!=1:raise RuntimeError('Original graph entry boundary changed')
 targets[0].keywords.append(ast.keyword(arg='captured_embedding_method_owner',value=ast.Name(id='safe_embed',ctx=ast.Load())))
 restored=copy.deepcopy(node)
 for x in ast.walk(restored):
  if isinstance(x,ast.Call):x.keywords[:]=[k for k in x.keywords if k.arg!='captured_embedding_method_owner']
 if ast.dump(restored,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('Original GPU operations/precision/control were changed')
 source=ast.unparse(ast.fix_missing_locations(node))+'\n';file='<prefix_constant_owner_lifetime_v10>'
 linecache.cache[file]=(len(source),None,source.splitlines(True),file);namespace=dict(fn.__globals__);exec(compile(source,file,'exec'),namespace);original.hidden=namespace[fn.__name__]
 RECEIPT=dict(original_hidden_AST_sha256=hashlib.sha256(ast.dump(before,include_attributes=False).encode()).hexdigest(),all_original_operators_and_precision_AST_exact=True,only_added_strong_reference_to_actual_constant_owner=True,new_attention_or_feature_values_never_fabricated=True)
 _INSTALLED=True;return RECEIPT
