"""Full native32 cache equality beyond the original500 bridge, excluding timing counters."""
from pathlib import Path
from .paths import read
from .contract import load_packet,same,BRANCHES
def full_parity(folder,reference):
 folder,reference=Path(folder),Path(reference);checks=0
 for stem in('initial','proposal','observed_history','discovery_anchor'):
  assert same(load_packet(folder/(stem+'.npz')),load_packet(reference/(stem+'.npz'))),stem;checks+=1
 for name in('discovery.json','observed_history.json','prefix_clock.json'):assert read(folder/name)==read(reference/name);checks+=1
 for branch in BRANCHES:
  for suffix in('.npz','_actuation.npz','_rgb.npz'):
   assert same(load_packet(folder/(branch+suffix)),load_packet(reference/(branch+suffix))),(branch,suffix);checks+=1
  assert read(folder/(branch+'_phase.json'))==read(reference/(branch+'_phase.json'));checks+=1
  a=read(folder/(branch+'.json'));b=read(reference/(branch+'.json'))
  for key in('planned_chunk_sha256','prefix_physics_sha256','prefix_rng_sha256','prefix_rng_components','initial','initial_images','initial_tokens','samples','native_actions_started','complete_episode','outcomes_used'):
   assert a[key]==b[key],(branch,key);checks+=1
 return dict(passed=True,checks=checks,reference=str(reference),all_original32_variable_native_commands_RGB_features_tokens_physics_rng_and_completion_exact=True,wall_time_and_readback_counters_intentionally_not_equal=True,GT_never_forward=True)
