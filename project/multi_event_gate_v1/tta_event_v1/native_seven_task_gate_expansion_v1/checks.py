"""Actual CPU weight compatibility and canonical language templates, no CUDA."""
import itertools,json,re,time
from pathlib import Path
from .contract import T,P,W,O,PLAIN_TASKS,GATED_TASKS,GATE_FREEZE,GATE_ADAPTER,read,sha,mapped,plan

def language_coverage(runtime):
    from native_pot_mic_semantics_v1.language import prompt,new_prompt
    from native_object_association_v1.semantic import prompt as previous
    result={};inputs={}
    categories={'lift_pot':'060_kitchenpot','handover_mic':'018_microphone'}
    for task,category in categories.items():
        file=runtime/'description/task_instruction'/(task+'.json');templates=read(file);inputs[str(file)]=sha(file)
        rows=[]
        for obj in sorted((runtime/'description/objects_description'/category).glob('*.json')):
            inputs[str(obj)]=sha(obj);descriptions=read(obj)
            for split in ('seen','unseen'):
                for template,description,(a,b) in itertools.product(templates[split],descriptions[split],(('left arm','right arm'),('right arm','left arm'))):
                    text=template.replace('{A}','the '+description).replace('{a}',a).replace('{b}',b)
                    assert not re.search(r'\{[^{}]+\}',text)
                    roles,phrases=prompt(text);assert roles['reference'] is None and roles['actual_object_count']==1
                    assert len(phrases)==1 and phrases[0] in text.lower()
                    rows.append(dict(split=split,model=obj.stem,instruction=text,phrases=phrases))
        result[task]=dict(expanded_variants=len(rows),all_expanded_variants_lexically_supported=True,records=rows)
    old=['Stack two bowls.','Place the red block on the green block.','Place the cup onto the coaster.','Place the shoe onto the mat.','Grab the hammer and beat the block.','Press the black switch with the left arm.','Grab the bottle.','Open the laptop lid.','Grab the roller with both arms.']
    for text in old:assert prompt(text)==previous(text)
    for text in ('Pass the microphone to the bottle.','Lift the pot beside the microphone.','Press the switch beside the microphone.'):
        assert not prompt(text)[1],text
    # A descriptor switch is not a second object during a microphone handover.
    assert new_prompt('Pass the microphone with slider switch to the other arm.')[0]['target']=='microphone'
    return dict(tasks=result,positive_old_languages_exact=len(old),negative_ambiguity_controls=3,actual_visual_coverage_NOT_tested=True,input_sha256=inputs)

def main():
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    p=plan();O.mkdir(parents=True,exist_ok=True)
    assert not (O/'CPU_compatibility_v1.json').exists()
    archive=read(GATE_FREEZE/'manifest.json');verified=[]
    for row in archive['files']:
        file=GATE_FREEZE/row['snapshot'];assert sha(file)==row['sha256'];verified.append(row['snapshot'])
    from native_visible_response_bridge_v1.bridge import load_model
    before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        response,q=load_model(mapped(p['current_TTA_qualification']))
        payload=torch.load(GATE_ADAPTER,map_location='cpu',weights_only=True)
    assert torch.equal(before,torch.random.get_rng_state())
    state=payload['model'];key='action_head.2.weight';arm=[0,1,2,3,4,5,7,8,9,10,11,12]
    new_weight=state[key][arm].detach().cpu();old=response.arm_weight.detach().cpu()
    assert new_weight.shape==old.shape==(12,64)
    _,singular,vh=torch.linalg.svd(new_weight.double(),full_matrices=False);projection=(vh.T@vh).to(new_weight.dtype)
    compatibility=dict(current_TTA_ARM_weight_equals_gate_head=torch.equal(old,new_weight),current_TTA_visible_projection_equals_gate_head=torch.equal(response.ARM_visible_projection,projection),maximum_ARM_weight_difference=float((old-new_weight).abs().max()),gate_ARM_rank=int(torch.linalg.matrix_rank(new_weight.double())),gate_head_alpha=.5,current_TTA_qualification_cannot_be_silently_reused_for_different_head=True,gate_hidden_semantics_must_be_revalidated=True)
    assert compatibility['gate_ARM_rank']==12
    language=language_coverage(Path(p['runtime_source']))
    (O/'language_coverage_v1.json').write_text(json.dumps(language,ensure_ascii=False,indent=2)+'\n')
    out=dict(passed=True,scope='CPU-only compatibility diagnosis, not official launch qualification or TTA utility',plan=p,compatibility=compatibility,archive_files_verified=len(verified),CPU_RNG_preserved=True,CUDA_initialized=False,language_coverage_file=str(O/'language_coverage_v1.json'),language_coverage_sha256=sha(O/'language_coverage_v1.json'),source_sha256={str(f):sha(f)for folder in (Path(__file__).parent,T/'native_pot_mic_semantics_v1')for f in folder.rglob('*')if f.is_file()})
    (O/'CPU_compatibility_v1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (O/'campaign_plan_v1.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,compatibility=compatibility,archive_files_verified=len(verified),language_variants={t:v['expanded_variants']for t,v in language['tasks'].items()},GPU_started=False),ensure_ascii=False))

if __name__=='__main__':main()
