from native_prepared_FIT_collection_v28.scope import transform as previous_transform,restore_runtime
from native_mixed_FIT_preparation_v1.prepare import rename,transform_extra
def transform(name,text):return transform_extra(name,rename(previous_transform(name,text)))
