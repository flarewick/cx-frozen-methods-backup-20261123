from pathlib import Path
from native_policy_execution_v1.common import T,read,sha,write
C=Path(__file__).resolve().parent;O=T/'outputs/native_direction_visible_training_v1'
RECIPE=dict(seed=17,updates=1000,optimizer='AdamW',learning_rate=.0003,weight_decay=.0001,packets_per_task=8,packets_per_batch=16)
