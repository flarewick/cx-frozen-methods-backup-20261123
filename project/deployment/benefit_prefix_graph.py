"""Original auxiliary eager prefix kernels, replayed as a checked CUDA graph.

The frozen max-autotune base action sampler is unchanged. Tokenization, image
preprocessing, signed pooling, type checks and CPU feature conversion stay in
the original encode function. No activations/predictions are reused as data.
"""
import ast,inspect,textwrap,threading,weakref,types
import torch
_CACHE={}

def eager(m,imgs,im_masks,tokens,lang_mask):
 from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
 embeddings,pad,att=m.embed_prefix(imgs,im_masks,tokens,lang_mask)
 mask=m._prepare_attention_masks_4d(make_att_2d_masks(pad,att))
 pos=torch.cumsum(pad,dim=1)-1
 m.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
 (hidden,_),_=m.paligemma_with_expert.forward(attention_mask=mask,position_ids=pos,past_key_values=None,inputs_embeds=[embeddings,None],use_cache=False)
 return hidden

def capture_safe_embed(model):
 """Lift only the original data-independent all-zero CPU-list CUDA constant."""
 original=model.embed_prefix
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
 assignments=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and ast.unparse(n.value)=='torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)']
 if len(assignments)!=1:raise RuntimeError('Original attention constant source changed')
 constant={}
 def attention(values,device):
  if not values or any(type(v)is not int or v!=0 for v in values):raise RuntimeError('Attention mask no longer a data-independent zero constant')
  key=(len(values),str(device))
  if not constant:
   if torch.cuda.is_current_stream_capturing():raise RuntimeError('Attention constant must initialize outside capture')
   constant[key]=torch.tensor(values,dtype=torch.bool,device=device)
  if key not in constant:raise RuntimeError('Static attention shape/device changed')
  return constant[key]
 assignments[0].value=ast.parse('_prefix_attention_constant(att_masks,pad_masks.device)',mode='eval').body
 ns=dict(original.__func__.__globals__,_prefix_attention_constant=attention)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 return types.MethodType(ns[node.name],model)

def signature(values):
 return tuple((tuple(v.shape),tuple(v.stride()),str(v.dtype),str(v.device))for v in values)

def hidden(m,imgs,im_masks,tokens,lang_mask):
 values=[*imgs,*im_masks,tokens,lang_mask];n=len(imgs);k=id(m);sig=signature(values)
 if n!=3 or len(im_masks)!=3 or any(v.device.type!='cuda'for v in values):raise RuntimeError('Original three-camera CUDA batch only')
 if m.training:raise RuntimeError('Frozen eval prefix required')
 entry=_CACHE.get(k)
 if entry is None:
  original_embed=m.embed_prefix;safe_embed=capture_safe_embed(m)
  static=[torch.empty_strided(v.shape,v.stride(),dtype=v.dtype,device=v.device)for v in values]
  for dst,src in zip(static,values):dst.copy_(src)
  if signature(static)!=sig:raise RuntimeError('Static prefix stride/dtype differs')
  current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
  m.embed_prefix=safe_embed
  try:
   with torch.cuda.stream(stream):
    for _ in range(2):warm=eager(m,static[:n],static[n:2*n],static[-2],static[-1])
   current.wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):
    try:out=eager(m,static[:n],static[n:2*n],static[-2],static[-1])
    except BaseException:
     import traceback
     traceback.print_exc()
     raise
  finally:m.embed_prefix=original_embed
  entry=dict(model=weakref.ref(m),signature=sig,static=static,graph=graph,out=out,owner=threading.get_ident());_CACHE[k]=entry
 else:
  if entry['model']()is not m or entry['signature']!=sig or entry['owner']!=threading.get_ident():raise RuntimeError('Prefix model/device/shape/owner changed')
  for dst,src in zip(entry['static'],values):dst.copy_(src)
 m.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
 entry['graph'].replay()
 return entry['out']

def build_encode():
 from cx_tta.cache_joint_prefix import encode as original
 node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
 block=next(n for n in node.body if isinstance(n,ast.With));start=next(i for i,n in enumerate(block.body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='(embeddings, pad, att)');end=next(i for i,n in enumerate(block.body)if isinstance(n,ast.Assign)and ast.unparse(n.value).startswith('m.paligemma_with_expert.forward'))
 original_ops=block.body[start:end+1]
 ours=ast.parse(textwrap.dedent(inspect.getsource(eager))).body[0].body[1:-1]
 if [ast.dump(n,include_attributes=False)for n in original_ops]!=[ast.dump(n,include_attributes=False)for n in ours]:raise RuntimeError('Original prefix tensor operations changed')
 block.body[start:end+1]=ast.parse('hidden=_prefix_graph_hidden(m,imgs,im_masks,tokens,lang_mask)').body
 ns=dict(original.__globals__,_prefix_graph_hidden=hidden)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 return ns['encode']

def install():
 import task_adapters.policy as policy
 policy.encode=build_encode()
