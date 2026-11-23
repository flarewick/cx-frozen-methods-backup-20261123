"""Two short physical sessions share one original twelve-scene GPU budget."""
def accounting(group,index,*,previous_spec=None,previous_release=None,initial_failed=0.,retry_seconds=0.):
 assert group in('plain','gated')and type(index)is int and 0<=index<17
 logical=index//2;cap=28800 if group=='plain'and logical==0 else 14400
 if index%2:
  assert initial_failed==0 and previous_spec is not None and previous_release is not None
  p=previous_spec;r=previous_release
  assert p['group']==group and p['stage_index']==index-1 and p['logical_GPU_budget_index']==logical
  assert r['own_gpu_released']and not r['remaining_own_gpu_pids']
  if group=='plain'and index==1:
   assert p['version']=='native_seven_official_v5'and p['stage_total_GPU_budget_seconds']==14400 and retry_seconds>0
  else:assert p['stage_total_GPU_budget_seconds']==cap and retry_seconds==0
  assert r['cleanup_seconds_after_start']>=0
  inherited=round(p['inherited_stage_GPU_seconds']+r['cleanup_seconds_after_start']+retry_seconds,3)
 else:
  assert previous_spec is None and previous_release is None
  assert initial_failed==retry_seconds==0
  inherited=initial_failed
 assert 0<=inherited<cap,'No automatic budget extension or reset between physical halves'
 return dict(logical_GPU_budget_index=logical,physical_session_within_logical_budget=index%2,inherited_stage_GPU_seconds=inherited,stage_total_GPU_budget_seconds=cap)

def check():
 p=dict(group='plain',stage_index=0,logical_GPU_budget_index=0,version='native_seven_official_v5',stage_total_GPU_budget_seconds=14400,inherited_stage_GPU_seconds=6962.875)
 r=dict(own_gpu_released=True,remaining_own_gpu_pids=[],cleanup_seconds_after_start=3456.907)
 a=accounting('plain',1,previous_spec=p,previous_release=r,retry_seconds=3985.)
 assert a['inherited_stage_GPU_seconds']==14404.782 and a['stage_total_GPU_budget_seconds']==28800
 for group in('plain','gated'):
  for index in range(2 if group=='plain'else 0,16,2):
   first=accounting(group,index);prior=dict(first,group=group,stage_index=index,version='native_seven_official_v9')
   second=accounting(group,index+1,previous_spec=prior,previous_release=r)
   assert second['stage_total_GPU_budget_seconds']==14400 and second['inherited_stage_GPU_seconds']==3456.907
  assert accounting(group,16)['logical_GPU_budget_index']==8
 for mutate in('unreleased','wrong_stage','over_budget','missing_failed_time'):
  prior=dict(p);release=dict(r);retry=3985.
  if mutate=='unreleased':release['own_gpu_released']=False
  elif mutate=='wrong_stage':prior['stage_index']=2
  elif mutate=='over_budget':retry=20000
  else:retry=0
  try:accounting('plain',1,previous_spec=prior,previous_release=release,retry_seconds=retry)
  except AssertionError:pass
  else:raise AssertionError(mutate)
 return True
