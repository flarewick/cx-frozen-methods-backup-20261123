"""Capture original deterministic denoise kernels; outer sampler/noise untouched."""
import ast,inspect,textwrap,types,copy,threading

def capture_safe_suffix(model):
 import torch
 original=model.embed_suffix;node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0];before=copy.deepcopy(node)
 found=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and ast.unparse(n.value)=='torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)'];assert len(found)==1
 constants={}
 def attention(values,dtype,device):
  assert values==[1]+[0]*31,'Frozen pi05 suffix attention constant changed'
  key=(tuple(values),dtype,str(device))
  if key not in constants:
   assert not torch.cuda.is_current_stream_capturing(),'Initialize constant outside capture'
   constants[key]=torch.tensor(values,dtype=dtype,device=device)
  return constants[key]
 found[0].value=ast.parse('_suffix_attention_constant(att_masks,embs.dtype,embs.device)',mode='eval').body
 restored=copy.deepcopy(node)
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_suffix_attention_constant':n.value=ast.parse('torch.tensor(att_masks,dtype=embs.dtype,device=embs.device)',mode='eval').body
 assert ast.dump(restored)==ast.dump(before),'Original suffix tensor operations changed'
 ns=dict(original.__func__.__globals__,_suffix_attention_constant=attention);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns)
 return types.MethodType(ns[node.name],model)

def tensors(cache):return [*cache.key_cache,*cache.value_cache]
def signature(values):return tuple((tuple(x.shape),tuple(x.stride()),str(x.dtype),str(x.device))for x in values)
class DenoiseGraph:
 def __init__(self,model):
  self.model=model;self.original=model.denoise_step;self.entry=None;self.owner=threading.get_ident();self.calls=0
 def __call__(self,state,pad,cache,actions,timestep):
  import torch
  assert threading.get_ident()==self.owner and not self.model.training and self.model.pi05
  values=[state,pad,actions,timestep,*tensors(cache)];sig=signature(values);assert all(x.device.type=='cuda'for x in values)
  if self.entry is None:
   static_cache=copy.deepcopy(cache);static=[torch.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device)for x in values[:4]]
   static+=tensors(static_cache);assert signature(static)==sig
   for dst,src in zip(static,values):dst.copy_(src)
   original_suffix=self.model.embed_suffix;safe=capture_safe_suffix(self.model);current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current);self.model.embed_suffix=safe
   try:
    with torch.cuda.stream(stream):
     for _ in range(2):warm=self.original(static[0],static[1],static_cache,static[2],static[3])
    current.wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=self.original(static[0],static[1],static_cache,static[2],static[3])
   finally:self.model.embed_suffix=original_suffix
   self.entry=dict(signature=sig,static=static,cache=static_cache,graph=graph,out=out,captured_suffix_owner=safe,cache_type=type(cache),cache_seen_tokens=getattr(cache,'_seen_tokens',None))
  else:
   e=self.entry;assert e['signature']==sig and e['cache_type']is type(cache)and e['cache_seen_tokens']==getattr(cache,'_seen_tokens',None),'Frozen denoiser signature changed'
   for dst,src in zip(e['static'],values):dst.copy_(src)
  self.calls+=1;self.entry['graph'].replay();return self.entry['out']
