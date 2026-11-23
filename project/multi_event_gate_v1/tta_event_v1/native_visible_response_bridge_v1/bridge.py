"""Explicit typed action-visible factory/loader; inherited online bridge math."""
from pathlib import Path
import hashlib
import torch
from native_direction_training_v1.data import T,read,sha
from native_shared_response_v1.session import qualified,state_sha
from native_direction_visible_training_v1.factory import model_for
from native_action_visible_response_v1.model import ActionVisibleResponse
from native_trained_response_bridge_v1.bridge import DirectionBridge as OriginalDirectionBridge

def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def require_visible(model,q):
 if type(model)is not ActionVisibleResponse or q.get('model_kind')!=ActionVisibleResponse.model_kind:raise ValueError('Explicit action-visible model and qualification required')
 if q.get('ARM_visible_interface_version')!=1 or int(model.response.ARM_visible_interface_version)!=1:raise ValueError('Wrong immutable action-visible interface version')
 if tensor_sha(model.ARM_visible_projection)!=q['ARM_visible_projection_sha256']:raise ValueError('Changed ARM projection')
 for file in(Path(__file__),T/'native_action_visible_response_v1/model.py',T/'native_direction_visible_training_v1/factory.py'):
  if q['source_sha256'].get(str(file))!=sha(file):raise ValueError('Action-visible implementation missing or changed')
 return True

def load_model(file):
 file=Path(file).resolve()
 if not file.is_relative_to((T/'outputs').resolve()):raise ValueError('Isolated qualified weights required')
 q=read(file)
 if q.get('model_kind')!=ActionVisibleResponse.model_kind:raise ValueError('Unprojected manifest cannot qualify action-visible model')
 assert q['frozen']and q['actual_training_host_exit0']and not q['awaiting_actual_host_exit0']and q['actual_updates']==1000
 assert sha(q['training_host_exit_file'])==q['training_host_exit_sha256']and read(q['training_host_exit_file'])['exit_code']==0
 assert sha(q['training_checkpoint'])==q['training_checkpoint_sha256']and sha(q['training_result_file'])==q['training_result_sha256']
 result=read(q['training_result_file']);assert result['complete']and result['actual_updates']==1000 and result['model_kind']==q['model_kind']
 model,_,_=model_for(read(T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json'))
 payload=torch.load(q['training_checkpoint'],map_location='cpu',weights_only=True)
 assert payload['step']==1000 and payload['protocol_sha256']==q['training_protocol_sha256']
 model.response.load_state_dict(payload['response'],strict=True)
 assert set(payload['buffers'])==set(dict(model.named_buffers()))
 for name,value in model.named_buffers():assert torch.equal(value,payload['buffers'][name]),name
 assert state_sha(model.response.state_dict())==q['response_state_sha256']
 require_visible(model,q);qualified(model,file);return model.eval(),q

class DirectionBridge(OriginalDirectionBridge):
 def __init__(self,model,qualification,*,mode,queue=None):
  require_visible(model,read(qualification))
  super().__init__(model,qualification,mode=mode,queue=queue)
  self._fixed_before=self._fixed_witness()
 def _fixed_witness(self):
  model=self.session.model
  return [(kind,name,value.data_ptr(),value._version,str(value.dtype),tuple(value.shape))for kind,items in(('parameter',model.named_parameters()),('buffer',model.named_buffers()))for name,value in items]
 def finish(self):
  assert self._fixed_before==self._fixed_witness(),'Action-visible fixed model/buffers changed during adaptation'
  result=super().finish();result['action_visible_fixed_parameters_and_buffers_unchanged']=True
  return result
