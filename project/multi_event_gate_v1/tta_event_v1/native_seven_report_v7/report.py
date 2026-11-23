"""Report only seven completed paired seed0/100 tasks, no partial promotion."""
import csv,json,math
from pathlib import Path
T=Path(__file__).resolve().parents[1]

def main():
 from native_seven_official_v9.common import read,verify,sha
 from native_seven_official_v9.qualification import qualified_stage
 from native_seven_official_v9.configuration import GROUPS,bounds
 from native_seven_official_v9.adoption import folder as stage_folder
 final=T/'outputs/native_seven_campaign_v8/all_seven_official100_complete.json';v=read(final);verify(v)
 assert v['complete']and len(v['groups'])==7 and v['TTA_episodes']==v['paired_control_episodes']==700
 rows=[];groups={};inputs={str(final):sha(final)};whole=0
 for group,cfg in GROUPS.items():
  for number in range(17):
   folder=stage_folder(group,number);root,spec=qualified_stage(folder)
   assert spec['group']==group and spec['official_seed']==0 and spec['actual_seed_start']==100000
   audit=read(root/'CPU_episode_audit.json');assert audit['passed']and audit['original_zero_shadow_controls_exact']
   start,end=bounds(number)
   for task in cfg['tasks']:assert sorted(r['episode_index']for r in audit['records']if r['task']==task)==list(range(start,end))
   whole+=audit['actual_native_episodes']+audit['closed_expert_task_lifetimes']['whole_native_reference_controls']
   for r in audit['records']:
    assert r['no_policy_outcome_filter']
    rows.append(dict(task=r['task'],episode_index=r['episode_index'],actual_scene_seed=r['seed'],official_seed=0,method_group=group,control='M_event_full+all_event_v118_gate'if group=='gated'else'M_event_full',control_success=bool(r['full']),TTA_success=bool(r['TTA']),rescue=bool(r['rescue']),harm=bool(r['harm']),accepted_updates=int(r['branches']['TTA']['accepted_updates']),first_changed_query=r['first_changed_query'],semantic_available=bool(r['branches']['TTA']['semantic_available']),stage=spec['stage'],GPU_uuid=spec['required_gpu_uuid'],response_checkpoint_sha256=spec['candidate_checkpoint_sha256']))
   for p in(root/'protocol.json',root/'CPU_episode_audit.json',root/'terminal_audit.json',folder/'actual_host_wait_exit.json'):inputs[str(p)]=sha(p)
 assert len(rows)==700 and len({(r['task'],r['episode_index'])for r in rows})==700
 for task in v['groups']:
  rr=[r for r in rows if r['task']==task];assert len(rr)==100 and len({r['actual_scene_seed']for r in rr})==100
  full=sum(r['control_success']for r in rr);tta=sum(r['TTA_success']for r in rr);rescue=sum(r['rescue']for r in rr);harm=sum(r['harm']for r in rr);discordant=rescue+harm
  assert tta-full==rescue-harm and full==v['groups'][task]['control_successes']and tta==v['groups'][task]['TTA_successes']
  p=1. if not discordant else min(1.,2*sum(math.comb(discordant,k)for k in range(min(rescue,harm)+1))/2**discordant)
  groups[task]=dict(control=rr[0]['control'],scenes=100,control_successes=full,TTA_successes=tta,difference_percentage_points=tta-full,rescues=rescue,harms=harm,semantic_available=sum(r['semantic_available']for r in rr),accepted_update_scenes=sum(r['accepted_updates']>0 for r in rr),changed_action_scenes=sum(r['first_changed_query']is not None for r in rr),paired_exact_p_unadjusted=p)
 out=T/'outputs/native_seven_report_v7';out.mkdir(exist_ok=False)
 with(out/'paired_episodes.csv').open('x',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['task'],r['episode_index'])))
 value=dict(complete=True,official_seed=0,groups=groups,primary_TTA_episodes=700,primary_control_episodes=700,actual_whole_episodes_including_controls=whole,statistical_test='Exact paired McNemar two-sided, exploratory p unadjusted for7 tasks',input_sha256=inputs)
 (out/'results.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 lines=['# 七任务官方 seed0 配对100景结果','', '|任务|同景控制|控制成功率|TTA成功率|差值|挽救／损失|','|---|---|---:|---:|---:|---:|']
 for task,g in groups.items():lines.append(f"|{task}|{g['control']}|{g['control_successes']}%|{g['TTA_successes']}%|{g['difference_percentage_points']:+d} pp|{g['rescues']}／{g['harms']}|")
 lines+=['','每任务分母100，包含缺反馈、零更新和失败回合。700个TTA与700个控制严格同场景配对；全部实际退出、原生动作、零／影子控制及资源审计通过后才纳入。','', '普通组使用当前78f1849响应候选和原stacking头；组合组使用冻结shared250/alpha0.5/v118门控，并通过原合格辅助头加入fast增量。与历史TTA093和不同底座结果分别保存。','',f'合格批次含全部控制与工程参照实际完成{whole}个完整回合。逐景记录见 paired_episodes.csv，模型、阶段和证明摘要见 results.json。']
 (out/'results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,ensure_ascii=False))
if __name__=='__main__':main()
