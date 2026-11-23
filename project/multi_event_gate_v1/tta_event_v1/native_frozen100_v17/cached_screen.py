"""Replay the exact completed expert prefix, then continue the original screen."""
from .common import write
def schedule_class(original):
 class CachedSchedule(original):
  def __init__(self,c,prepared,spec,root,original_planner):
   super().__init__(c,prepared,spec,root,original_planner)
   self.cached=(spec.get('cached_preparation')or{}).get('task_cache',{});self.case_keys=(spec.get('cached_preparation')or{}).get('case_request_keys',{})
   self.case_diagnostics=(spec.get('cached_preparation')or{}).get('case_diagnostics',{});self.cached_drained=set();self.cached_positions={task:0 for task in spec['tasks']};self.cached_items={task:0 for task in spec['tasks']}
  def consume(self,name,until=None):
   entry=self.cached[name];state=self.state[name]
   while self.cached_positions[name]<len(entry['rows']):
    row=entry['rows'][self.cached_positions[name]]
    assert row['seed']==state['next_seed']and row['episode_index']==state['accepted']
    if until is not None and row['seed']>until:break
    write(self.root/'screening'/name/('%d.json'%row['seed']),row);state['rows'].append(row);state['next_seed']+=1
    if row['eligible']:state['accepted']+=1
    self.cached_positions[name]+=1
  def next(self,name):
   with self.locks[name]:
    entry=self.cached.get(name)if name not in self.cached_drained else None
    if entry is not None:
     index=self.cached_items[name]
     if index<len(entry['items']):
      item=entry['items'][index];self.consume(name,item['seed']);assert self.state[name]['accepted']==item['episode_index']+1
      self.cached_items[name]+=1;write(self.root/'initial_prefetch'/(item['case_id']+'.json'),dict(content_key=self.case_keys[item['case_id']],cached_initial_input_from_verified_prefix=True,actual_policy_initial_byte_match_still_required=True,diagnostic_identity=self.case_diagnostics.get(item['case_id'])));write(self.root/'accepted'/(item['case_id']+'.json'),item);return item
     self.consume(name);assert all(self.state[name][k]==v for k,v in entry['end_state'].items());self.cached_drained.add(name)
   return super().next(name)
 return CachedSchedule
