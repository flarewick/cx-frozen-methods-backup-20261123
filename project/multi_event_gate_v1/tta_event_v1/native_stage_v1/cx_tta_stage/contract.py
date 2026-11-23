"""Whole scene origin-measurement packets, explicit varying length and censoring."""
from pathlib import Path
import numpy as np
from .paths import read,write,output,digest,identity
from .views import VIEWS
TICKS=(32,64,125,187,250,375,500)
BRANCHES=('Z_reference','C_original','Z_sham')
def case_branches(folder):return BRANCHES

def load_packet(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a)
def validate_packet(p,ticks=None):
 assert set(p)=={'ticks','elapsed_s','joint','visual','command_state','commands','dt_s'} and all(np.isfinite(v).all() for v in p.values())
 ts=p['ticks'];dt=p['dt_s'];n=len(ts);count=len(dt);assert 1<=count<=5000 and n>=1 and ts.dtype==np.int64 and np.all(np.diff(ts)>0) and ts[0]>0 and ts[-1]==count
 if ticks is not None:assert np.array_equal(ts,ticks)
 assert dt.shape==(count,) and np.all(dt>0) and np.all(dt==dt[0]) and abs(float(dt[0])-.004)<1e-8
 assert p['commands'].shape==(count,14) and p['joint'].shape==(n,3,28672) and p['visual'].shape==(n,3,3,2,2,32) and p['command_state'].shape==(n,14)
 assert np.array_equal(p['elapsed_s'],np.cumsum(dt)[ts-1])

def verify_case(folder,projection=None):
 from cx_tta.eval_diagnostics import array_hash
 from .query import compose
 folder=Path(folder)
 if (folder/'unavailable.json').exists():
  v=read(folder/'unavailable.json');assert v['reason'] in ('retained_source_unavailable','no_event_within_four_queries','native_prefix_censored','zero_native_steps')
  return dict(available=False,complete_case_record=True,reason=v['reason'],effect_used=False)
 proposal=load_packet(folder/'proposal.npz');initial=load_packet(folder/'initial.npz');packets={};meta={};traces={};phases={}
 for branch in BRANCHES:
  p=load_packet(folder/(branch+'.npz'));validate_packet(p);m=read(folder/(branch+'.json'));t=load_packet(folder/(branch+'_actuation.npz'));ph=read(folder/(branch+'_phase.json'));rgb=load_packet(folder/(branch+'_rgb.npz'));count=len(p['dt_s']);n=len(p['ticks'])
  assert digest(folder/(branch+'.npz'))==m['file_sha256'] and digest(folder/(branch+'_actuation.npz'))==m['actuation_sha256'] and m['planned_chunk_sha256']==array_hash(compose(proposal['base'],proposal['raw_delta'],branch))
  assert m['view_order']==list(VIEWS) and not m['complete_episode'] and not m['outcomes_used']
  assert np.array_equal(t['dt_s'],p['dt_s']) and t['arm_position_target'].shape==(count,12) and t['arm_velocity_target'].shape==(count,12) and t['gripper_logical_command'].shape==(count,2)
  assert np.array_equal(t['arm_position_target'].astype(np.float32),p['commands'][:,[0,1,2,3,4,5,7,8,9,10,11,12]]) and np.array_equal(t['gripper_logical_command'].astype(np.float32),p['commands'][:,[6,13]])
  for key in ('gripper_position_target','gripper_velocity_target','gripper_joint_counts'):assert len(t[key])==count and np.isfinite(t[key]).all()
  assert ph['origin']['committed_ticks']==count and ph['receipt']['arrived_tick']==count and not ph['receipt']['pending'] and len(ph['readouts'])==n
  if ph['origin']['complete_origin']:assert ph['origin']['reason']=='origin_chunk_completed' and ph['origin']['commands_returned']==32
  for cam in range(3):
   assert rgb['initial_'+str(cam)].dtype==np.uint8 and rgb['samples_'+str(cam)].shape==(n,*rgb['initial_'+str(cam)].shape)
   assert array_hash(rgb['initial_'+str(cam)])==m['initial_images'][str(cam)]
   for i,(tick,row) in enumerate(zip(p['ticks'],m['samples'])):assert row['tick']==tick and array_hash(rgb['samples_'+str(cam)][i])==row['view_images'][0][str(cam)]
  for row in ph['readouts']:
   for request in row['stage_requests']:
    assert request['decision_tick']==request['observation_tick']==row['tick'] and request['actual_lag_s']+1e-9>=request['requested_lag_s'] and request['lateness_s']<=float(p['dt_s'][0])+1e-8 and request['origin_chunk']=='origin'
  assert len(ph['receipt']['closed'])==1
  for key in initial:assert m['initial'][key]==array_hash(initial[key])
  packets[branch]=p;meta[branch]=m;traces[branch]=t;phases[branch]=ph
 assert same(packets['Z_reference'],packets['Z_sham']) and same(traces['Z_reference'],traces['Z_sham']) and phases['Z_reference']==phases['Z_sham'] and meta['Z_reference']['samples']==meta['Z_sham']['samples']
 for key in ('prefix_physics_sha256','prefix_rng_sha256','prefix_rng_components','initial','initial_images','initial_tokens'):assert all(v[key]==meta['Z_reference'][key] for v in meta.values())
 if projection is not None:
  from cx_tta.structured_features import project
  pre=load_packet(projection)
  for p in packets.values():
   v,_=project(p['joint'].reshape(-1,28672),pre);assert np.array_equal(v.reshape(p['visual'].shape),p['visual'])
 from .prefix_time import verify_prefix_clock
 verify_prefix_clock(folder)
 bridge=read(folder/'prefix_compatibility.json');assert bridge['passed']
 return dict(available=True,complete=True,whole_origin_measurements=True,exact_sham=True,same_prefix=True,legacy500_bridge=True,variable_duration_explicit=True,phase_after_actual_execution=True,original32commands_no_hold_filler=True)

def dataset(spec,gpu_uuid):
 from .paths import OUTPUT
 m=dict(protocol_identity=spec['identity'],gpu_uuid=gpu_uuid,encoder_batch=3,scene_order=[v['case_id'] for v in spec['scenes']]);m['identity']=identity(m);root=output(OUTPUT/'collections'/m['identity'][:24]);root.mkdir(parents=True,exist_ok=True)
 f=root/'manifest.json'
 if f.exists():assert read(f)==m
 else:write(f,m)
 return root,m

def load_journal(root,manifest,spec):
 f=Path(root)/'journal.json';j=read(f) if f.exists() else dict(identity=manifest['identity'],cases=[]);assert j['identity']==manifest['identity'] and len(j['cases'])<=len(spec['scenes'])
 for i,r in enumerate(j['cases']):
  assert r['case_id']==spec['scenes'][i]['case_id']
  for name,h in r['evidence_sha256'].items():assert digest(output(Path(root)/name))==h
  assert verify_case(Path(root)/r['folder'],spec['projection'])==r['audit']
 return j

def commit(root,manifest,spec,j,folder,session):
 audit=verify_case(folder,spec['projection']);i=len(j['cases']);r=dict(split=spec['scenes'][i]['split'],case_id=spec['scenes'][i]['case_id'],folder=str(Path(folder).relative_to(root)),session=session,audit=audit,evidence_sha256={str(p.relative_to(root)):digest(p) for p in sorted(Path(folder).iterdir()) if p.is_file()});write(Path(root)/'journal.json',dict(identity=manifest['identity'],cases=[*j['cases'],r]));j['cases'].append(r);return r
