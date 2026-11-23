"""Strict first500 bridge against original whole developer packet, inclextra reads."""
from pathlib import Path
import numpy as np
from .paths import read
from .contract import load_packet,same,BRANCHES

def parity(folder,reference):
 folder,reference=Path(folder),Path(reference);checks=0
 for file in ('initial','proposal','observed_history','discovery_anchor'):
  assert same(load_packet(folder/(file+'.npz')),load_packet(reference/(file+'.npz'))),file;checks+=1
 for file in ('discovery.json','observed_history.json','prefix_clock.json'):assert read(folder/file)==read(reference/file);checks+=1
 for branch in BRANCHES:
  old=load_packet(reference/(branch+'.npz'));new=load_packet(folder/(branch+'.npz'));assert len(new['dt_s'])>=500;lookup={int(t):i for i,t in enumerate(new['ticks'])};ids=[lookup[int(t)] for t in old['ticks']]
  for key in old:
   got=new[key][:500] if key in ('commands','dt_s') else new[key][ids]
   assert got.dtype==old[key].dtype and got.shape==old[key].shape and got.tobytes()==old[key].tobytes(),(branch,key);checks+=1
  a=load_packet(folder/(branch+'_actuation.npz'));b=load_packet(reference/(branch+'_actuation.npz'));assert all(a[k][:500].tobytes()==v.tobytes() for k,v in b.items());checks+=len(b)
  a=load_packet(folder/(branch+'_rgb.npz'));b=load_packet(reference/(branch+'_rgb.npz'))
  for key,v in b.items():assert (a[key][ids] if key.startswith('samples_') else a[key]).tobytes()==v.tobytes(),(branch,key);checks+=1
  a=read(folder/(branch+'.json'));b=read(reference/(branch+'.json'))
  for key in ('prefix_physics_sha256','prefix_rng_sha256','prefix_rng_components','initial','initial_images','initial_tokens','planned_chunk_sha256'):assert a[key]==b[key],(branch,key);checks+=1
  for newrow,oldrow in zip([a['samples'][i] for i in ids],b['samples']):assert newrow==oldrow;checks+=1
 return dict(passed=True,legacy_first500_raw_RGB_physics_rng_tokens_commands_exact=True,extra_passive_stage_reads_do_not_change_legacy_observations=True,checks=checks,reference=str(reference),full_origin_not_claimed_same_length=True)
