from pathlib import Path
import json,numpy as np,h5py
from native_best100_resume_v7.common import P,mapped,read
manifest=read(P/'outputs/base_cache_20260926_024808_872087/manifest.json')
for task in('place_empty_cup','place_shoe'):
 row=next(r for r in manifest['files']if r['task']==task and mapped(r['path']).exists()and mapped(r['source']).exists())
 with np.load(mapped(row['path']),allow_pickle=False)as z:print('NPZ',task,{k:(str(z[k].dtype),z[k].shape)for k in z.files})
 print('SOURCE',row['source'],'INSTRUCTION',row['instruction'])
 with h5py.File(mapped(row['source']),'r')as f:
  def show(name,obj):
   if isinstance(obj,h5py.Dataset):print(name,obj.shape,str(obj.dtype))
  f.visititems(show)
