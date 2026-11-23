"""Language-only category-pair compatibility; frozen detector math unchanged."""
import re,numpy as np
from native_cpu_seed_v1.seed import semantic as original
from native_transfer_semantics_v1.semantic import prompt as previous_prompt
local=original.local;nms=original.nms;iou=original.iou;rgb_sha=original.rgb_sha;text_sha=original.text_sha

def prompt(language):
 roles,phrases=previous_prompt(language)
 if phrases:return roles,phrases
 text=language.lower()
 source=list(re.finditer(r'\b(stapler|mouse|fan|bread|loaf)\b',text))
 references=list(re.finditer(r'\b(mat|pad|skillet|pan)\b',text))
 if not source or not references or not re.search(r'\b(place|put|set|drop|position|move|transfer|lay|grab|pick|take|lift)\b',text):return roles,[]
 relations=list(re.finditer(r'\b(on|onto|in|into|inside|over|above|to)\b',text))
 canonical=lambda name:'bread'if name=='loaf'else 'mat'if name=='pad'else 'skillet'if name=='pan'else name
 valid=[(s,r)for s in source for r in references if s.start()<r.start()and any(s.end()<=p.start()<r.start()for p in relations)]
 pairs={(canonical(s.group(1)),canonical(r.group(1)))for s,r in valid}
 if len(pairs)!=1:return roles,[]
 names=list(next(iter(pairs)))
 return dict(directed=True,target=names[0],reference=names[1],kind='explicit_heterogeneous_placement_pair',interface_version='native_transfer_semantics_v2'),names

class FrozenInstances(original.FrozenInstances):
 def infer(self,images,languages):
  from PIL import Image
  torch=self.torch
  if len(images)!=len(languages) or not images:raise ValueError('Matched legal images/language required')
  roles_phrases=[prompt(v) for v in languages]
  if any(not p for _,p in roles_phrases):raise ValueError('Unsupported language remains explicit')
  texts=['. '.join(p)+'.' for _,p in roles_phrases];pil=[Image.fromarray(local.rgb_check(im)) for im in images]
  inputs=self.processor(images=pil,text=texts,return_tensors='pt',padding=True).to(self.device)
  with torch.inference_mode():outputs=self.detector(**inputs)
  probabilities=outputs.logits.sigmoid().cpu().numpy();pred=outputs.pred_boxes.cpu().numpy();results=[];selected=[];active=[]
  for i,(rgb,language,(roles,phrases),text) in enumerate(zip(images,languages,roles_phrases,texts)):
   encoded=self.processor.tokenizer(text,return_offsets_mapping=True);offsets=encoded['offset_mapping'];assert encoded['input_ids']==inputs.input_ids[i,:len(encoded['input_ids'])].cpu().tolist()
   h,w=rgb.shape[:2];xy=pred[i].copy();boxes=np.c_[xy[:,:2]-xy[:,2:]/2,xy[:,:2]+xy[:,2:]/2]*[w,h,w,h];boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
   candidates=[];cursor=0
   for phrase in phrases:
    start=text.index(phrase,cursor);end=start+len(phrase);cursor=end;ids=[j for j,(a,b) in enumerate(offsets) if b>a and a>=start and b<=end]
    assert ids;score=probabilities[i][:,ids].min(axis=1);eligible=np.flatnonzero(score>=.3);kept=nms(boxes[eligible],score[eligible]);candidates.append([dict(box=boxes[eligible[k]].tolist(),score=float(score[eligible[k]]),phrase=phrase) for k in kept])
   reason=None;chosen=[]
   if len(phrases)==2:
    for cc in candidates:
     if not cc or len(cc)>1 and cc[0]['score']-cc[1]['score']<.08:reason='missing_or_ambiguous_text_grounding';break
     chosen.append(cc[0])
    if len(chosen)==2 and iou(chosen[0]['box'],chosen[1]['box'])>.5:reason='both_phrases_ground_to_same_instance'
   else:
    cc=candidates[0]
    if len(cc)<2 or len(cc)>2 and cc[1]['score']-cc[2]['score']<.1:reason='missing_or_ambiguous_two_instances'
    else:chosen=sorted(cc[:2],key=lambda x:x['box'][0])
   value=dict(image_sha256=rgb_sha(rgb),instruction_sha256=text_sha(language),roles=roles,text_prompt=text,candidates=candidates,available=False,reason=reason,
    source='FrozenGroundingDINO-base+SAM-vit-base, float32/eager/nativeTransformers4.53.2, image+languageonly',semantic_correctness_verified=False)
   if reason is None and len(chosen)==2:active.append(i);selected.append([v['box'] for v in chosen]);value['selected']=chosen
   results.append(value)
  del outputs,inputs
  if active:
   ins=self.sam_processor(images=[pil[i] for i in active],input_boxes=selected,return_tensors='pt').to(self.device)
   with torch.inference_mode():out=self.sam(**ins,multimask_output=False)
   masks=self.sam_processor.image_processor.post_process_masks(out.pred_masks.cpu(),ins['original_sizes'].cpu(),ins['reshaped_input_sizes'].cpu())
   scores=out.iou_scores.cpu().numpy()
   for k,i in enumerate(active):
    mm=masks[k][:,0].numpy().astype(bool);assert mm.shape==(2,*images[i].shape[:2]);area=mm.reshape(2,-1).sum(1);overlap=int((mm[0]&mm[1]).sum());union=int((mm[0]|mm[1]).sum())
    distinct=overlap/max(union,1)<=.2 and overlap/max(int(area.min()),1)<=.35
    enough=bool(np.all(area>=32) and np.all(scores[k,:,0]>=.7));r=results[i];r.update(mask_scores=scores[k,:,0].tolist(),mask_area=area.tolist(),mask_overlap_iou=overlap/max(union,1),distinct_instances=bool(distinct))
    if distinct and enough:r.update(available=True,reason='text_grounded_distinct_masks',masks=mm)
    else:r['reason']='overlapping_or_weak_instance_masks'
  return results
