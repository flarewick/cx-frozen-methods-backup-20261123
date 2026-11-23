"""Original DINO/SAM math, with explicit one- or two-real-object adjudication."""
import numpy as np
from native_transfer_semantics_v2.semantic import FrozenInstances as Previous,iou,nms,rgb_sha,text_sha,local
from .language import prompt,VERSION
class FrozenInstances(Previous):
 def infer(self,images,languages):
  from native_transfer_semantics_v2.semantic import prompt as old_prompt
  if len(images)!=len(languages)or not images:raise ValueError('Matched actual RGB/language required')
  values=[]
  for image,language in zip(images,languages):
   if old_prompt(language)[1]:values.append(super().infer([image],[language])[0])
   else:values.append(self._infer_new(image,language))
  return values
 def _infer_new(self,image,language):
  from PIL import Image
  roles,phrases=prompt(language)
  if not phrases:raise ValueError('Unsupported language must remain explicit missing')
  count=len(phrases)
  if count not in(1,2):raise ValueError('One or two real objects only')
  text='. '.join(phrases)+'.';pil=Image.fromarray(local.rgb_check(image));torch=self.torch
  inputs=self.processor(images=[pil],text=[text],return_tensors='pt',padding=True).to(self.device)
  with torch.inference_mode():outputs=self.detector(**inputs)
  probability=outputs.logits.sigmoid().cpu().numpy()[0];pred=outputs.pred_boxes.cpu().numpy()[0]
  encoded=self.processor.tokenizer(text,return_offsets_mapping=True);offsets=encoded['offset_mapping'];assert encoded['input_ids']==inputs.input_ids[0,:len(encoded['input_ids'])].cpu().tolist()
  h,w=image.shape[:2];xy=pred.copy();boxes=np.c_[xy[:,:2]-xy[:,2:]/2,xy[:,:2]+xy[:,2:]/2]*[w,h,w,h];boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
  candidates=[];chosen=[];cursor=0;reason=None
  for phrase in phrases:
   start=text.index(phrase,cursor);end=start+len(phrase);cursor=end;ids=[j for j,(a,b)in enumerate(offsets)if b>a and a>=start and b<=end];assert ids
   score=probability[:,ids].min(axis=1);eligible=np.flatnonzero(score>=.3);kept=nms(boxes[eligible],score[eligible]);cc=[dict(box=boxes[eligible[k]].tolist(),score=float(score[eligible[k]]),phrase=phrase)for k in kept];candidates.append(cc)
   if not cc or len(cc)>1 and cc[0]['score']-cc[1]['score']<.08:reason='missing_or_ambiguous_text_grounding'
   else:chosen.append(cc[0])
  if count==2 and len(chosen)==2 and iou(chosen[0]['box'],chosen[1]['box'])>.5:reason='both_phrases_ground_to_same_instance'
  reference_missing_reason=None
  target=candidates[0]
  target_known=bool(target)and(len(target)==1 or target[0]['score']-target[1]['score']>=.08)
  if count==2 and target_known and reason is not None:
   reference_missing_reason=reason;count=1;chosen=[target[0]];reason=None
   roles=dict(roles,declared_reference=roles['reference'],reference=None,directed=False,actual_object_count=1,absent_reference_explicit=True,kind='observed_target_requested_reference_unknown')
  result=dict(image_sha256=rgb_sha(image),instruction_sha256=text_sha(language),roles=roles,text_prompt=text,candidates=candidates,available=False,reason=reason,interface_version=VERSION,actual_object_count=count,reference_missing_reason=reference_missing_reason,actual_object_slots=[True,count==2],source='FrozenGroundingDINO+SAM; same weights,float32,thresholds; actual RGB+language only',semantic_correctness_verified=False)
  del outputs,inputs
  if reason is not None:return result
  assert len(chosen)==count;result['selected']=chosen
  ins=self.sam_processor(images=[pil],input_boxes=[[x['box']for x in chosen]],return_tensors='pt').to(self.device)
  with torch.inference_mode():out=self.sam(**ins,multimask_output=False)
  masks=self.sam_processor.image_processor.post_process_masks(out.pred_masks.cpu(),ins['original_sizes'].cpu(),ins['reshaped_input_sizes'].cpu())[0][:,0].numpy().astype(bool);quality=out.iou_scores.cpu().numpy()[0,:,0];assert masks.shape==(count,h,w)
  area=masks.reshape(count,-1).sum(1);enough=bool(np.all(area>=32)and np.all(quality>=.7));distinct=True;overlap=union=0
  if count==2:
   overlap=int((masks[0]&masks[1]).sum());union=int((masks[0]|masks[1]).sum());distinct=overlap/max(union,1)<=.2 and overlap/max(int(area.min()),1)<=.35
  result.update(mask_scores=quality.tolist(),mask_area=area.tolist(),mask_overlap_iou=overlap/max(union,1),distinct_instances=bool(distinct))
  if not(enough and distinct):result['reason']='overlapping_or_weak_instance_masks';return result
  slots=np.zeros((2,h,w),bool);slots[:count]=masks;assert count==2 or not slots[1].any()
  result.update(available=True,reason='text_grounded_real_objects',masks=slots,absent_reference_is_empty_not_second_object=count==1)
  return result
