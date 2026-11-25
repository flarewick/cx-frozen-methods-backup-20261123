"""Execute the original official instruction selection, including RNG consumption."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
from .common import C
B=C/'multibase_v1';R=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin')
source=R/'description/utils/generate_episode_instructions.py'
spec=importlib.util.spec_from_file_location('rwc_original_language',source);language=importlib.util.module_from_spec(spec);spec.loader.exec_module(language)
inputs=json.loads((B/'protocols/official_language_inputs.json').read_text())['records']
def consume(case):
 row=inputs[case['task']+':'+str(case['seed'])]
 values=language.generate_episode_descriptions(case['task'],[row['scene_info']],row['descriptions'])
 selected=str(np.random.choice(values[0][row['instruction_type']]))
 assert selected==case['prompt'],('Official generated instruction differs from frozen reference',case['task'],case['seed'],selected,case['prompt'])
 return selected
