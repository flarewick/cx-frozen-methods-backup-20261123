"""Same original batch3 encoder; replace only eager prefix dispatch by checked graph."""
import ast,inspect,textwrap,copy

def build():
 from cx_tta_feedback_confirm5.views import batched_encode as original
 from benefit_prefix_graph_lifetime_v10 import install
 install()
 from benefit_prefix_graph import hidden,eager
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
 block=next(n for n in node.body if isinstance(n,ast.With));start=next(i for i,n in enumerate(block.body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='(embeddings, pad, att)');end=next(i for i,n in enumerate(block.body)if isinstance(n,ast.Assign)and ast.unparse(n.value).startswith('m.paligemma_with_expert.forward'))
 ops=copy.deepcopy(block.body[start:end+1]);reference=ast.parse(textwrap.dedent(inspect.getsource(eager))).body[0].body[1:-1]
 assert [ast.dump(n)for n in ops]==[ast.dump(n)for n in reference],'Original batch3 prefix tensor operators changed'
 block.body[start:end+1]=ast.parse('hidden=_checked_graph_hidden(m,imgs,im_masks,tokens,lang_mask)').body
 namespace=dict(original.__globals__,_checked_graph_hidden=hidden)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
 return namespace[original.__name__]
