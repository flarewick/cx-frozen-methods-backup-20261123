"""Exact reversible runtime scheduling additions; original numerical code remains."""
IMPORT=" from native_early_next_preparation_v1.schedule import ready as early_next_ready,isolated as isolated_next,build_prepare as next_prepare,adopt_closed_tasks\n"
ORIGINAL="""   from .future_preparation import prepare as prepare_next
   with schedule.serial_screen_lock.background():
    prepare_next(schedule,c,root,spec,observe,install_render,discard_observation)
"""
INDEPENDENT="""   from .future_preparation import prepare as prepare_next
   private=isolated_next(schedule,OriginalSchedule.next,Lane)
   prepared['fixed_next_schedule']=private
   prepare_next=next_prepare(prepare_next)
   with schedule.serial_screen_lock.background():
    prepare_next(private,c,root,spec,observe,install_render,discard_observation)
"""
ADOPT="  if 'fixed_next_schedule'in prepared:c.call(schedule.lanes[spec['tasks'][0]],adopt_closed_tasks,schedule,prepared['fixed_next_schedule'])\n"

def transform(source):
 assert IMPORT not in source and source.count(ORIGINAL)==1
 marker=" from .cache import CacheMaintenance\n";assert source.count(marker)==1
 changed=source.replace(marker,IMPORT+marker)
 old='while not future_ready.wait(.05):';assert changed.count(old)==1
 changed=changed.replace(old,'while not early_next_ready(work,spec,future_ready):')
 changed=changed.replace(ORIGINAL,INDEPENDENT)
 marker="  assert future_started[0];future_thread.join(timeout=3);assert not future_thread.is_alive()\n"
 assert changed.count(marker)==1;changed=changed.replace(marker,marker+ADOPT)
 assert restore(changed)==source
 return changed

def restore(source):
 for old,new in [(IMPORT,''),('while not early_next_ready(work,spec,future_ready):','while not future_ready.wait(.05):'),(INDEPENDENT,ORIGINAL),(ADOPT,'')]:
  assert source.count(old)==1,old
  source=source.replace(old,new)
 return source
