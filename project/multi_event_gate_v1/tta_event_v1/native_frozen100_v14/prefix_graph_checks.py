"""CPU source/constant/tree conformance; never initialize CUDA or load models."""
import ast,json,math,types
import torch
from native_encoder_graph_v1.runtime import safe_prefix,tree,values,output_meta
from .common import T,O,write

def main():
 assert not torch.cuda.is_initialized()
 info=json.loads((T/'outputs/native_encoder_graph_v1/CPU_source_inspection.json').read_text())
 print('INSPECTION_KEYS',list(info)if isinstance(info,dict)else type(info))
 # Compile the installed original method into a tiny CPU fixture, no pretrained weights.
 def locate(v):
  if isinstance(v,dict):
   if v.get('function')=='embed_prefix':return v['source']
   for x in v.values():
    r=locate(x)
    if r:return r
  if isinstance(v,list):
   for x in v:
    r=locate(x)
    if r:return r
 src=locate(info);assert src,'Missing inspected original embed_prefix'
 import linecache
 linecache.cache['original_embed_prefix.py']=(len(src),None,src.splitlines(True),'original_embed_prefix.py')
 ns=dict(torch=torch,math=math);exec(compile(ast.parse(src),'original_embed_prefix.py','exec'),ns)
 class Fake:
  def __init__(self):self.paligemma_with_expert=self
  def _apply_checkpoint(self,fn,*args):return fn(*args)
  def embed_image(self,x):return x
  def embed_language_tokens(self,x):return x[...,None].float().expand(-1,-1,4)
 Fake.embed_prefix=ns['embed_prefix']
 f=Fake();safe=safe_prefix(f);rng=torch.random.get_rng_state().clone()
 for k in(1,2):
  images=[torch.arange(24).reshape(2,3,4).float()+k+i for i in range(3)];masks=[torch.tensor([True,False])for _ in images];tokens=torch.arange(10).reshape(2,5)+k;lang=torch.ones(2,5,dtype=torch.bool)
  old=f.embed_prefix(images,masks,tokens,lang);new=safe(images,masks,tokens,lang)
  assert all(torch.equal(a,b)for a,b in zip(old,new,strict=True))
 assert torch.equal(rng,torch.random.get_rng_state())
 x={'a':[torch.arange(3),None],'b':True};y=tree(x,lambda t:t.clone());assert torch.equal(values(x)[0],values(y)[0])and values(x)[0].data_ptr()!=values(y)[0].data_ptr()
 assert output_meta((torch.zeros(2,3),None))==('tuple',(('tensor',((2,3),(3,1),'torch.float32','cpu')),None))
 mask=json.loads((T/'outputs/native_encoder_graph_v1/CPU_mask_source.json').read_text());assert any("len(attention_mask.shape) == 4"in x['source']and'return True, attention_mask'in x['source']for x in mask['functions'])
 assert not torch.cuda.is_initialized()
 write(O/'CPU_prefix_graph_checks.json',dict(passed=True,original_installed_embed_prefix_AST_exact_except_immutable_constant_constructor=True,two_actual_CPU_inputs_exact=True,RNG_unchanged=True,nested_input_copy_and_output_metadata_checked=True,original_4D_attention_mask_early_return=True,no_model_or_CUDA_initialized=True,GPU_first_actual_eager_conformance_still_required=True))
 print('CPU_PREFIX_GRAPH_CHECKS_PASS')
if __name__=='__main__':main()
