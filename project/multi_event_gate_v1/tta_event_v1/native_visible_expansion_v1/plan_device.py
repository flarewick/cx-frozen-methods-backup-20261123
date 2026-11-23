"""Fresh resource-only device assignment before cohort source freeze/preload."""
from pathlib import Path
import argparse,json,sys
T=Path(__file__).resolve().parents[1];P=T.parents[1]
sys.path.insert(0,str(T));sys.path.insert(0,str(P/'deployment'))
from native_visible_expansion_v1.placement import plan
from native_resource_policy_v4.admission import require_admission
from native_fast_inventory_v1.inventory import inventory
import gpu_doctor

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=list('abc'),required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 O=T/'outputs'/('native_visible_expand_'+a.cohort+'_v1');assert not(O/'protocol.json').exists()and not(O/'device_plan.json').exists()
 v=plan(a.gpu);fleet=inventory();require_admission(fleet,v['gpu_uuid']);info=gpu_doctor.gpu_snapshot(a.gpu)
 for name,value in [('cpu_placement_plan.json',v),('device_plan.json',dict(gpu=a.gpu,gpu_uuid=v['gpu_uuid'],snapshot=info,fresh_fleet=fleet,resource_assignment_only=True,fixed_population_and_candidate_unchanged=True,design_GPU_indices_are_preferences=True))]:
  with(O/name).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print('EXPANDED_COHORT_FRESH_DEVICE_PLAN',a.cohort,a.gpu,v['selected_logical_CPUs'],flush=True)
if __name__=='__main__':main()
