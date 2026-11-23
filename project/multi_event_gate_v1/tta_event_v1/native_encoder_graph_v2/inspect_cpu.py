from pathlib import Path
import ast,json,hashlib
T=Path(__file__).resolve().parents[1]
def main():
 rt=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/policy/pi05/src/openpi/models_pytorch')
 sources=[(rt/'pi0_pytorch.py',{'embed_prefix'}),(rt/'gemma_pytorch.py',{'forward'}),(Path('/opt/cx_venv/lib/python3.11/site-packages/transformers/models/gemma/modeling_gemma.py'),{'forward','_update_causal_mask'})]
 report=[]
 for p,names in sources:
  text=p.read_text();tree=ast.parse(text)
  for cls in tree.body:
   if not isinstance(cls,ast.ClassDef):continue
   if p.name=='modeling_gemma.py'and cls.name!='GemmaModel':continue
   for fn in cls.body:
    if isinstance(fn,ast.FunctionDef)and fn.name in names:
     source=ast.get_source_segment(text,fn);report.append(dict(file=str(p),file_sha256=hashlib.sha256(text.encode()).hexdigest(),class_name=cls.name,function=fn.name,source=source));print('SOURCE',str(p),cls.name,fn.name);print(source)
 out=T/'outputs/native_encoder_graph_v1';out.mkdir(parents=True,exist_ok=True)
 with(out/'CPU_source_inspection.json').open('x')as f:json.dump(dict(CPU_only=True,no_model_execution=True,sources=report),f,indent=2);f.write('\n')
if __name__=='__main__':main()
