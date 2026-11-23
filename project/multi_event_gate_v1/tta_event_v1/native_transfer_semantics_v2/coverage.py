"""Enumerate official language variants without native scene/outcome access."""
from pathlib import Path
from types import SimpleNamespace
import ast,re,json,itertools,hashlib
T=Path(__file__).resolve().parents[1];R=T.parents[1]/'outputs/evaluations/structured_action/beat_block_hammer/L0/eval_20261003_164626_759241/robotwin'
def fn(path,name,scope):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]
def main():
 roles=fn(T/'stage_feedback_v2/association.py','language_roles',dict(re=re,COLORS=('red','green','blue','yellow')))
 original=fn(T/'language_instances_v2/semantic.py','prompt',dict(re=re,local=SimpleNamespace(language_roles=roles)))
 old=fn(T/'native_transfer_semantics_v1/semantic.py','prompt',dict(re=re,original=SimpleNamespace(prompt=original)))
 new=fn(T/'native_transfer_semantics_v2/semantic.py','prompt',dict(re=re,previous_prompt=old))
 def descriptions(model,ids):
  out=[]
  for i in ids:
   v=json.loads((R/'description/objects_description'/model/('base'+str(i)+'.json')).read_text())
   out.extend('the '+x for x in(v.get('unseen')or v['seen']))
  return out
 colors=['Red','Green','Blue','Yellow','Cyan','Magenta','Black','Gray','Orange','Purple','Brown','Pink','Lime','Olive','Teal','Maroon','Navy','Coral','Turquoise','Indigo','Beige','Tan','Silver']
 configs={'move_stapler_pad':dict(A=descriptions('048_stapler',range(7)),B=colors[:8]),'place_mouse_pad':dict(A=descriptions('047_mouse',range(3)),B=colors[:8]),'place_fan':dict(A=descriptions('099_fan',[4,5]),B=colors),'place_bread_skillet':dict(A=descriptions('106_skillet',range(4)),B=descriptions('075_bread',[0,1,3,5,6]))}
 result={}
 for task,params in configs.items():
  params=dict(params,a=['the left arm','the right arm']);values=json.loads((R/'description/task_instruction'/(task+'.json')).read_text())
  groups={}
  for split in ('unseen','seen'):
   total=supported=0;missing=[];old_exact=0
   for template in values[split]:
    keys=set(re.findall(r'{([^}]+)}',template))
    if keys not in(set(params),set(params)-{'a'}):continue
    for row in itertools.product(*(params[k]for k in sorted(keys))):
     text=template
     for k,v in zip(sorted(keys),row):text=text.replace('{'+k+'}',v)
     a,b=old(text),new(text);total+=1;supported+=bool(b[1])
     if a[1]:assert a==b;old_exact+=1
     if not b[1]and len(missing)<5:missing.append(text)
   assert total>0
   # Formal evaluator uses only official unseen instructions.
   if split=='unseen':assert supported==total,(task,total,supported,missing)
   groups[split]=dict(total=total,supported=supported,unsupported_examples=missing,old_supported_outputs_exact=old_exact)
  result[task]=groups
 for text in ['Place the cup onto the coaster','Place the shoe on the mat','Stack the red block on the blue block','Stack the bowls']:
  assert old(text)==new(text)
 for text in ['Grab the roller','Turn the switch','Place the mat on the fan','Put the bread beside the skillet','Place the mouse on the mat or in the skillet']:
  assert not new(text)[1],text
 out=dict(passed=True,tasks=result,old_paths_exact=True,unsupported_negative_controls_passed=True,no_native_scene_or_success_data_used=True,no_CUDA=True,scope='Language eligibility only; detector coverage and TTA efficacy unknown')
 folder=T/'outputs/native_transfer_semantics_v2';folder.mkdir(parents=True,exist_ok=True)
 with(folder/'CPU_coverage.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
