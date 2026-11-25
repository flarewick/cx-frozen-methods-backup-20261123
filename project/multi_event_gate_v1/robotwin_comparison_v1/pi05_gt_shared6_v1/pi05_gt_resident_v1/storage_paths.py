from pathlib import Path
import json
from .common import O
ROOT=O
def output_path(p):
 p=Path(p).resolve();assert p!=O.resolve()and p.is_relative_to(O.resolve()),'Output must stay in the new native_runs scope';return p
def write_json(p,x):
 p=output_path(p)
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
