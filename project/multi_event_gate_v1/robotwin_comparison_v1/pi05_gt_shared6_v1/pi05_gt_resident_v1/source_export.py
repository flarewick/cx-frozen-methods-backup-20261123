"""Causal source serialization; optional relative diagnostics have one fixed dtype."""
from pathlib import Path
import numpy as np
from .common import O,write,sha
FIELDS=('context','command','target','start_points','end_points','point_alive','relative_target','relative_available')
def arrays(rows):
 if not rows:return {}
 values={k:np.stack([r[k]for r in rows])for k in FIELDS}
 # The original per-row subtraction is unchanged. Its serialized dtype must not
 # depend on a later, not-yet-observed missing-partner NaN elsewhere in a stream.
 values['relative_target']=values['relative_target'].astype(np.float64)
 for name in ('command_query','start_tick','end_tick'):values[name]=np.asarray([r[name]for r in rows],np.int64)
 return values

def save_source(folder,summary):
 folder=Path(folder).resolve();assert folder.is_relative_to(O.resolve());folder.mkdir(parents=True,exist_ok=False)
 metadata={k:v for k,v in summary.items()if k not in ('rows','mask_readouts')};local=arrays(summary['rows']);metadata['local_file']=None
 if local:
  file=folder/'local_feedback.npz'
  with file.open('xb')as f:np.savez_compressed(f,**local)
  metadata.update(local_file=str(file),local_sha256=sha(file))
 masks=summary['mask_readouts'];metadata['mask_file']=None
 if masks:
  file=folder/'arrived_masks.npz'
  with file.open('xb')as f:np.savez_compressed(f,masks=np.stack([r['masks']for r in masks]),presence=np.stack([r['presence']for r in masks]),global_ticks=np.asarray([r['global_tick']for r in masks],np.int64))
  metadata.update(mask_file=str(file),mask_sha256=sha(file))
 metadata['rows']=[{k:r[k]for k in ('object_index','start_tick','end_tick','segment','dt_s','initial_points','matched_points','command_query')}for r in summary['rows']]
 metadata['mask_readouts']=[{k:v for k,v in r.items()if k not in ('masks','presence')}for r in masks]
 metadata['relative_diagnostic_serialization_dtype']='float64'
 metadata['primary_model_inputs_not_changed_by_serialization']=True
 write(folder/'source.json',metadata);return metadata
