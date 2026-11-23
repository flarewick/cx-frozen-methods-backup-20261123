"""Strict validation-only handling of an absent/present proven inactive diagnostic alias."""
import copy,json,collections
from pathlib import Path
import numpy as np
import benefit_followup_protocol_v2 as p
import benefit_growth_canonical_audit as canonical
BASE_EXACT=canonical.exact
INSTALLED=False
DUMP_ACTORS=collections.Counter({'063_tabletrashbin':1,'garbage':5,'ground':1,'wall':1,'table':1,'011_dustbin':1})
LIFT_ACTORS=collections.Counter({'ground':1,'wall':1,'table':1})

def prove(rows):
 poses=[];present=[];snapshots=[];kind=None
 for row in rows:
  for side in('physical_before','physical_after'):
   snap=row['offline_only'][side];actors=collections.Counter(v['name']for v in snap['actors']);arts=collections.Counter(v['name']for v in snap['articulations'])
   if actors==LIFT_ACTORS and arts==collections.Counter({'':1,'060_kitchenpot':1}):current='lift_pot_without_native_clutter'
   elif actors==DUMP_ACTORS and len(snap['articulations'])==1:current='dump_bin_bigbin_without_native_clutter'
   else:raise AssertionError('Only two explicit no-clutter native inventories are permitted')
   assert snap['usable']and not snap['coverage']['errors']and snap['coverage']['actor_count']==len(snap['actors'])and snap['coverage']['articulation_count']==len(snap['articulations'])
   assert kind in(None,current);kind=current;snapshots.append(snap)
  for action in row['offline_only']['actions']:
   for side in('before','after'):
    dictionary=action['offline_only'][side]['state']['offline_only']['object_poses'];has='cluttered_obj'in dictionary;present.append(has)
    if has:poses.append(dictionary['cluttered_obj'])
 assert present and(all(present)or not any(present)),'Alias presence cannot change within the recorded episode'
 if poses:
  ghost=poses[0];assert len(ghost)==7 and np.isfinite(np.asarray(ghost,np.float64)).all()and all(value==ghost for value in poses),'Present alias must be finite and constant in every recorded action'
  for snap in snapshots:
   active=[v['pose']for v in snap['actors']]+[v['root_pose']for v in snap['articulations']]
   assert ghost not in active,'Never remove any current native object'
 return dict(native_inventory=kind,native_snapshots=len(snapshots),diagnostic_action_snapshots=len(present),alias_present=bool(poses),alias_pose=poses[0]if poses else None,alias_all_actions_constant_if_present=True,alias_absent_from_all_current_native_snapshots=True,native_coverage_error_free=True)

def cleaned(row):
 value=copy.deepcopy(row)
 for action in value['offline_only']['actions']:
  for side in('before','after'):action['offline_only'][side]['state']['offline_only']['object_poses'].pop('cluttered_obj',None)
 return value

def compare(a,b,limit=None,base=BASE_EXACT,corrections=None):
 issues=base(a,b,limit)
 if not issues or any(k!='full_passive_trace'for k,q in issues):return issues
 try:
  pa,pb=prove(a[2]),prove(b[2]);assert pa['native_inventory']==pb['native_inventory']and(pa['alias_present']or pb['alias_present'])
 except(AssertionError,KeyError,TypeError,ValueError):return issues
 ca=list(a);cb=list(b);ca[2]=[cleaned(v)for v in a[2]];cb[2]=[cleaned(v)for v in b[2]]
 if canonical.ORIGINAL_EXACT(ca,cb,limit):return issues
 if corrections is not None:corrections.append(dict(validation_extension='supplement_proven_inactive_alias_v27',seed=a[0]['seed'],reference_branch=a[0]['pilot_branch'],candidate_branch=b[0]['pilot_branch'],reference_session=a[0]['process_session'],candidate_session=b[0]['process_session'],prefix_limit=limit,raw_differences=issues,reference_inactive_alias=pa,candidate_inactive_alias=pb,all_other_native_contact_legal_action_and_clock_values_original_zero_tolerance=True,helper_sha256=p.sha(__file__)))
 return []

def install():
 global INSTALLED
 if INSTALLED:return
 canonical.verify_amendment();amendment=p.HOME/'supplement_v24/inactive_alias_amendment_v27.json';value=json.loads(amendment.read_text());proof=p.HOME/'supplement_v24/inactive_alias_CPU_proof_v27.json';checked=json.loads(proof.read_text());assert checked['passed']and checked['helper_sha256']==p.sha(__file__)and value['helper_sha256']==p.sha(__file__)and value['CPU_proof_sha256']==p.sha(proof)and value['original_source_contract_sha256']==canonical.source_contract()
 def exact(a,b,limit=None):return compare(a,b,limit,base=BASE_EXACT,corrections=canonical.CORRECTIONS)
 canonical.exact=exact;INSTALLED=True
