from native_prepared_FIT_collection_v27.scope import transform as previous_transform, restore_runtime
from native_postexit_compat_v1.prepare_successor import rename, host, plan, NOTE
def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='host.py':text=host(text)
 if name=='collection_plan.py':text=plan(text)
 if name=='AGENTS.md':text+=NOTE
 return text
