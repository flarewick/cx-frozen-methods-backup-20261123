"""Whole-case immutable recovery; incomplete trajectories never enter this dataset."""
from pathlib import Path
import shutil
from .paths import read,write,sha

def materialize(spec,root):
 records=[]
 for record in spec['recovered_cases']:
  index=record['index'];source=Path(record['capture_file']).parent;target=root/('case%03d'%index)
  copied={}
  for key,h in record['source_files_sha256'].items():
   p=Path(key);assert sha(p)==h;dest=target/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert sha(dest)==h,'Existing recovered file differs; never overwrite'
   else:
    with p.open('rb')as src,dest.open('xb')as out:shutil.copyfileobj(src,out,8*1024*1024)
   assert sha(dest)==h;copied[str(dest)]=h
  assert read(target/'capture.json')==read(record['capture_file'])
  records.append(dict(index=index,case_id=record['case_id'],source_capture=record['capture_file'],source_capture_sha256=record['capture_sha256'],destination_sha256=copied))
 result=dict(complete=True,rows=records,recovered_terminal_cases=28,recovered_native_available=27,new_GPU_cases=68,
   original_JSON_paths_preserved_as_readonly_provenance=True,incomplete_queries_never_spliced=True,source_normal_exit_NOT_claimed=True)
 f=root/'reused_cases.json'
 if f.exists():assert read(f)==result
 else:write(f,result)
 return result

def recovered_rows(spec,root):
 value=read(root/'reused_cases.json');assert value['complete']and value['recovered_terminal_cases']==28
 rows={}
 for record in value['rows']:
  for p,h in record['destination_sha256'].items():assert sha(p)==h
  row=read(root/('case%03d'%record['index'])/'capture.json');assert row==read(record['source_capture'])
  assert row['case_id']==spec['scenes'][record['index']]['case_id']and row['index']==record['index'];rows[record['index']]=row
 assert len(rows)==28 and set(rows).isdisjoint(spec['remaining_new_indices'])and set(rows)|set(spec['remaining_new_indices'])==set(range(96));return rows
