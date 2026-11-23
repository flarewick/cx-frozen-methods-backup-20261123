from native_prepared_FIT_collection_v29.scope import transform as previous_transform,restore_runtime
from native_fixed_FIT_inputs_v5.prepare import rename,extra
def transform(name,text):return extra(name,rename(previous_transform(name,text)))
