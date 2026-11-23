from pathlib import Path
import itertools,json
from native_best100_resume_v7.common import read,mapped,T
from .semantic import prompt
runtime=mapped(read(T/'outputs/native_best100_resume_v7/study.json')['runtime_source'])
for task in('place_empty_cup','place_shoe'):
 templates=read(runtime/'description/task_instruction'/(task+'.json'))['unseen'];strings=[]
 if task=='place_empty_cup':
  for p,a,b in itertools.product(templates,read(runtime/'description/objects_description/021_cup/base0.json')['unseen'],read(runtime/'description/objects_description/019_coaster/base0.json')['unseen']):strings.append(p.replace('{A}','the '+a).replace('{B}','the '+b))
 else:
  for number in range(10):
   for p,a,arm in itertools.product(templates,read(runtime/'description/objects_description/041_shoe'/('base%d.json'%number))['unseen'],('left','right')):strings.append(p.replace('{A}','the '+a).replace('{a}','the '+arm+' arm'))
 bad=[s for s in strings if not prompt(s)[1]];print(task,'TOTAL',len(strings),'MISSING',len(bad));print('\n'.join(bad[:18]))
