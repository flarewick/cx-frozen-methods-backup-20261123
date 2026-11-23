"""All52 real RGB streams through new live callbacks; saved core, CPU only."""
from pathlib import Path
import importlib.util,json,uuid
from types import SimpleNamespace
import numpy as np
import torch,cv2
from cx_tta_feedback_confirm5 import live_feedback as live
from cx_tta_feedback_confirm5.live_audit import verify_live
from cx_tta_feedback_confirm5.paths import OUTPUT as O,write,digest
from native_delayed_query_v1.primitives import configuration
from native_delayed_query_v1.predictor import FrozenPredictor
from native_delayed_query_v1.mask_replay import SavedReadouts
C=Path(__file__).resolve().parent;T=C.parent
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
class Coordinator:
    cpu_only_fixture=True
    def call(self,lane,fn,*a,**k):return fn(*a,**k)
class Cache:
    epoch=1
    def __init__(self):self.calls=0
    def record(self,kind,epoch):assert kind=='tracker'and epoch==1;self.calls+=1

def run():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    spec=read(T/'outputs/native_arrived_mask_v3/protocol.json')
    teacher=T/'outputs/native_arrived_mask_v3/collections/79fa25d4427ec9cd476539a2'
    records=read(teacher/'GPU_work_complete.json')['rows']
    sp=importlib.util.spec_from_file_location('live_replay_only_data',T/'native_arrived_mask_v3/data.py')
    d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
    geometry=configuration();predictor=FrozenPredictor();coordinator=Coordinator()
    root=O/'cpu_fixtures'/uuid.uuid4().hex;root.mkdir(parents=True)
    original=live.OffloadedCore;checks=0;outputs=[]
    try:
        for job,record in zip(spec['jobs'],records):
            data=d.LoadedStream(job);history=data.history;ticks=data.clock['query_observation_ticks']
            rows=[dict(rgb=tuple(history['cam'+str(k)][i]for k in range(3)),command_state=history['states'][i])for i in range(len(ticks))]
            live.OffloadedCore=lambda model,record=record:SavedReadouts(record)
            cache=Cache();obj=live.LiveFeedback(coordinator,None,None,job['earliest_seed'],data.seed_masks(),rows,ticks,
                                               job['instruction'],geometry,predictor,cache)
            folder=root/('stream%03d'%job['index']);folder.mkdir()
            (folder/'observed_history.npz').symlink_to(job['history'])
            segments={s['kind']:s for s in job['stream']['segments']};active=None
            for images,meta in data.native():
                seg=segments[meta['kind']]
                if active!=meta['kind']:
                    active=meta['kind'];trace=load(seg['actuation']);packet=load(seg['packet'])
                    callback=obj.capture_callback(active,seg['absolute_start_tick'],len(ticks)-1+(active=='following'))
                    stem=job['branch']+'_'+active
                    for suffix in('.npz','.json','_actuation.npz','_rgb.npz','_receipt.json'):
                        source=Path(seg['packet']).parent/(stem+suffix)
                        (folder/(stem+suffix)).symlink_to(source)
                frame=next(f for f in job['stream']['frames']if f['global_tick']==meta['global_tick'])
                tick=meta['global_tick']-seg['absolute_start_tick']
                recorder=SimpleNamespace(rows=[{k:v[i]for k,v in trace.items()if k!='dt_s'}for i in range(tick)],dt=trace['dt_s'][:tick].tolist())
                observed=dict(rgb=images,command_state=packet['command_state'][frame['index']],observation_tick=meta['global_tick'])
                callback(tick,float(np.sum(trace['dt_s'][:tick])),observed,recorder)
                if meta['global_tick']==seg['annotation_available_tick']and active=='origin':
                    receipt=read(seg['receipt']);obj.close_origin(dict(completion=receipt['completion'],packet=packet,
                        origin=receipt['origin'],commands_returned=receipt['commands_returned']))
                    if job['stream']['following_query_sampled']:
                        obj.reference_boundary(SimpleNamespace(take_action_cnt=0,step_lim=1000,eval_success=False),observed,query=len(ticks))
            obj.export(folder,job['branch'])
            item=dict(case_id='CPU_fixture_'+job['case_id'],instruction=job['instruction'],earliest_seed=job['earliest_seed'],
                      teacher_records={job['branch']:str(teacher/'streams'/('stream%03d'%job['index'])/'record.json')})
            audit=verify_live(folder,job['branch'],item,cpu_only_fixture=True)
            assert cache.calls==audit['actual_tracking_calls'];checks+=1+audit['checks']
            outputs.append(dict(index=job['index'],audit=audit))
            print('LIVE_CALLBACK_PARENT_REPLAY',job['index'],audit['rows'],flush=True)
    finally:live.OffloadedCore=original
    assert not torch.cuda.is_initialized();checks+=1
    report=dict(passed=True,checks=checks,streams=len(outputs),rows=sum(r['audit']['rows']for r in outputs),results=outputs,
        fixture_root=str(root),CPU_fixture_only=True,saved_core_substitute=True,actual_recorded_q1_replayed=True,
        full_native_terminal_permission_NOT_proven=True,actual_third_query=False,
        source_sha256={str(p):digest(p)for p in(C/'live_replay_checks.py',C/'cx_tta_feedback_confirm5/live_feedback.py',C/'cx_tta_feedback_confirm5/live_audit.py')},
        scope='All52 actual RGB/command streams through new callbacks/queue and independent parent reconstruction. Saved core and fixture continuation permission, no real SAM2/actor/CUDA/physics/thirdquery proof.',
        GPU_ready=False,online_updates=0,TTA_utility_proven=False)
    write(O/'live_replay_checks.json',report)
    print('ALL52_LIVE_CALLBACK_PARENT_CPU_REPLAY_PASS',checks,report['rows'],flush=True)
if __name__=='__main__':run()
