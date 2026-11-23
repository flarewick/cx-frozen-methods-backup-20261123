from pathlib import Path
import ast,json,hashlib
T=Path(__file__).resolve().parents[1]
p=Path('/opt/cx_venv/lib/python3.11/site-packages/transformers/masking_utils.py');text=p.read_text();tree=ast.parse(text)
result=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef)and n.name in('create_causal_mask','_preprocess_mask_arguments'):
  source=ast.get_source_segment(text,n);result.append(dict(function=n.name,source=source));print(n.name);print(source[:10000])
out=T/'outputs/native_encoder_graph_v1/CPU_mask_source.json'
with out.open('x')as f:json.dump(dict(file=str(p),file_sha256=hashlib.sha256(text.encode()).hexdigest(),functions=result,no_GPU_or_model_execution=True),f,indent=2)
