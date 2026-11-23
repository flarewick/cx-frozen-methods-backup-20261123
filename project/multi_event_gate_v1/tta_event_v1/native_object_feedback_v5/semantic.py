"""Noun-first preservation, descriptor adjudication, honest missing references."""
import copy,numpy as np
from native_object_feedback_v4.semantic import FrozenInstances as Descriptive
from native_object_feedback_v3.semantic import FrozenInstances as Noun
from native_transfer_semantics_v2.semantic import prompt as old_prompt
from native_object_feedback_v3.language import new_prompt as noun_prompt
from native_object_feedback_v4.language import new_prompt as descriptive_prompt
from .language import prompt,VERSION
class FrozenInstances(Descriptive):
 def tag(self,value,path):
  value['interface_version']=VERSION;value['roles']=dict(value['roles'],interface_version=VERSION);value['grounding_decision_path']=path;return value
 def target_only(self,image,value):
  candidates=value.get('candidates',[])
  if len(candidates)!=2 or not candidates[0]:return None
  main=candidates[0]
  if len(main)>1 and main[0]['score']-main[1]['score']<.08:return None
  from PIL import Image
  torch=self.torch;pil=Image.fromarray(image);ins=self.sam_processor(images=[pil],input_boxes=[[main[0]['box']]],return_tensors='pt').to(self.device)
  with torch.inference_mode():out=self.sam(**ins,multimask_output=False)
  masks=self.sam_processor.image_processor.post_process_masks(out.pred_masks.cpu(),ins['original_sizes'].cpu(),ins['reshaped_input_sizes'].cpu())[0][:,0].numpy().astype(bool);quality=out.iou_scores.cpu().numpy()[0,:,0]
  assert masks.shape==(1,*image.shape[:2]);area=masks.reshape(1,-1).sum(1)
  if area[0]<32 or quality[0]<.7:return None
  result=copy.deepcopy(value);slots=np.zeros((2,*image.shape[:2]),bool);slots[0]=masks[0];roles=result['roles'];result['roles']=dict(roles,declared_reference=roles['reference'],reference=None,directed=False,actual_object_count=1,absent_reference_explicit=True,kind='observed_target_requested_reference_unknown')
  result.update(available=True,reason='text_grounded_target_reference_missing',reference_missing_reason=value['reason'],actual_object_count=1,actual_object_slots=[True,False],selected=[main[0]],mask_area=area.tolist(),mask_scores=quality.tolist(),distinct_instances=True,mask_overlap_iou=0.,masks=slots,absent_reference_is_empty_not_second_object=True)
  return result
 def infer(self,images,languages):
  if len(images)!=len(languages)or not images:raise ValueError('Matched actual RGB/language required')
  results=[]
  for image,language in zip(images,languages):
   if old_prompt(language)[1]:results.extend(super().infer([image],[language]));continue
   noun=Noun._infer_new(self,image,language)
   if noun['available']:results.append(self.tag(noun,'original_unambiguous_noun_retained'));continue
   if descriptive_prompt(language)[1]!=noun_prompt(language)[1]:
    descriptor=Descriptive._infer_new(self,image,language)
    if descriptor['available']:results.append(self.tag(descriptor,'actual_instruction_descriptor_adjudication'));continue
   single=self.target_only(image,noun)
   results.append(self.tag(single if single is not None else noun,'explicit_target_only_reference_unknown'if single is not None else'unknown_retained'))
  return results
