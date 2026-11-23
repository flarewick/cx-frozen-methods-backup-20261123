"""Original prefix operators captured once on actual inputs; exact eager conformance."""
import ast,copy,inspect,textwrap,types,threading

def safe_prefix(model):
 import torch
 original=model.embed_prefix;node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0];before=copy.deepcopy(node)
 found=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and ast.unparse(n.value)=='torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)'];assert len(found)==1
 constants={}
 def attention(values,dtype,device):
  assert all(v==0 for v in values)
  key=(tuple(values),dtype,str(device),torch.is_inference_mode_enabled())
  if key not in constants:
   assert device.type!='cuda'or not torch.cuda.is_current_stream_capturing()
   constants[key]=torch.tensor(values,dtype=dtype,device=device)
  return constants[key]
 found[0].value.func=ast.Name(id='_prefix_attention_constant',ctx=ast.Load())
 restored=copy.deepcopy(node)
 for n in ast.walk(restored):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_prefix_attention_constant':n.func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='tensor',ctx=ast.Load())
 assert ast.dump(restored)==ast.dump(before)
 ns=dict(original.__func__.__globals__,_prefix_attention_constant=attention)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 return types.MethodType(ns[node.name],model)

def tree(x,fn):
 import torch
 if isinstance(x,torch.Tensor):return fn(x)
 if isinstance(x,list):return [tree(v,fn)for v in x]
 if isinstance(x,tuple):return tuple(tree(v,fn)for v in x)
 if isinstance(x,dict):return {k:tree(v,fn)for k,v in x.items()}
 assert x is None or isinstance(x,(bool,int,float,str)),type(x)
 return x

def values(x):
 import torch
 if isinstance(x,torch.Tensor):return [x]
 if isinstance(x,(list,tuple)):return [t for v in x for t in values(v)]
 if isinstance(x,dict):return [t for v in x.values()for t in values(v)]
 if x is None or isinstance(x,(bool,int,float,str)):return []
 assert hasattr(x,'key_cache')and hasattr(x,'value_cache'),type(x)
 return [*x.key_cache,*x.value_cache]

def tensor_sig(x):return (tuple(x.shape),tuple(x.stride()),str(x.dtype),str(x.device))
def output_meta(x):
 import torch
 if isinstance(x,torch.Tensor):return ('tensor',tensor_sig(x))
 if isinstance(x,(list,tuple)):return (type(x).__name__,tuple(output_meta(v)for v in x))
 if x is None:return None
 return (type(x).__module__,type(x).__name__,getattr(x,'_seen_tokens',None),tuple(tensor_sig(t)for t in values(x)))

class ActualGraph:
 def __init__(self,fn,name,owner,maximum=2):
  self.original=fn;self.signature=inspect.signature(fn);self.name=name;self.owner=owner;self.entries={};self.maximum=maximum
 def __call__(self,*args,**kwargs):
  import torch
  assert threading.get_ident()==self.owner
  bound=self.signature.bind(*args,**kwargs);bound.apply_defaults();arguments=dict(bound.arguments)
  sig=repr(tree(arguments,tensor_sig))+repr((torch.is_grad_enabled(),torch.is_inference_mode_enabled(),torch.is_autocast_enabled('cuda'),str(torch.get_autocast_dtype('cuda'))))
  src=values(arguments);assert src and all(x.device.type=='cuda'for x in src)
  if sig not in self.entries:
   assert len(self.entries)<self.maximum,'Unexpected frozen prefix shape/context'
   cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
   reference=self.original(**arguments);reference_meta=output_meta(reference)
   static=tree(arguments,lambda x:torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device));dst=values(static)
   for d,s in zip(dst,src,strict=True):d.copy_(s)
   current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
   with torch.cuda.stream(stream):
    for _ in range(2):warm=self.original(**static)
   current.wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):out=self.original(**static)
   current.wait_stream(stream);graph.replay();torch.cuda.synchronize()
   assert output_meta(out)==reference_meta,'Prefix graph output structure changed'
   pairs=list(zip(values(out),values(reference),strict=True));assert pairs and all(torch.equal(a,b)for a,b in pairs),'Original actual eager and captured prefix differ'
   assert torch.equal(cpu_rng,torch.random.get_rng_state())and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Prefix capture changed RNG'
   self.entries[sig]=dict(static=static,graph=graph,out=out,meta=reference_meta,calls=1,first_actual_eager_exact=True,output_tensor_count=len(pairs),batch=int(src[0].shape[0]))
   return reference
  e=self.entries[sig];assert output_meta(e['out'])==e['meta'],'Prefix cache was mutated downstream'
  for d,s in zip(values(e['static']),src,strict=True):d.copy_(s)
  e['graph'].replay();e['calls']+=1
  return e['out']

class PrefixGraphs:
 def __init__(self,model):
  self.owner=threading.get_ident();self.model=model;self.embed_original=model.embed_prefix;self.forward_original=model.paligemma_with_expert.forward
  self.embed=ActualGraph(safe_prefix(model),'embed_prefix',self.owner)
  self.prefix=ActualGraph(self.forward_original,'language_prefix',self.owner)
  model.embed_prefix=self.embed;model.paligemma_with_expert.forward=self.forward
 def forward(self,*args,**kwargs):
  b=self.prefix.signature.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
  if v['inputs_embeds'][0]is None:return self.forward_original(*args,**kwargs)
  assert v['inputs_embeds'][1]is None and v['past_key_values']is None
  return self.prefix(*args,**kwargs)
 def proof(self):
  rows={name:[{k:v for k,v in e.items()if k in('calls','first_actual_eager_exact','output_tensor_count','batch')}for e in obj.entries.values()]for name,obj in [('embed_prefix',self.embed),('language_prefix',self.prefix)]}
  assert all(len(v)==2 and all(e['calls']>1 and e['first_actual_eager_exact']for e in v)for v in rows.values())
  return dict(passed=True,entries=rows,original_operators_shapes_dtype_and_once_noise_unchanged=True,no_extra_policy_queries_or_updates=True,only_first_actual_input_bounded_capture_warmup=True)
