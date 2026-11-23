"""All52 saved streams, frame-at-a-time CPU adapter vs frozen batch evidence.

Teacher masks are provided externally; this does not prove an online tracker.
"""
from pathlib import Path
import importlib.util
import json
import os
import hashlib
import pickle
import time
import numpy as np
import cv2
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_delayed_query_v1'

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    environment=dict(os.environ);affinity=set(os.sched_getaffinity(0));threads=cv2.getNumThreads()
    rng=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()
    from native_delayed_query_v1.local_feedback import LocalFeedbackStream
    from native_delayed_query_v1.primitives import configuration,SOURCE_SHA256
    assert environment==dict(os.environ)and affinity==set(os.sched_getaffinity(0))
    assert threads==cv2.getNumThreads()and rng==hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()
    checks=3
    d=module('actual_native_arrived_loader_for_stream_parity',T/'native_arrived_mask_v3/data.py')
    cfg,cam,bots=configuration()
    local=read(T/'outputs/native_arrived_feedback_v1/local/protocol.json')
    a=load(T/'outputs/native_arrived_feedback_v1/local/data.npz')
    x=load(T/'outputs/native_local_response_v2/features.npz')['context']
    teacher=T/'outputs/native_arrived_mask_v3/collections/79fa25d4427ec9cd476539a2'
    jobs=read(T/'outputs/native_arrived_mask_v3/protocol.json')['jobs']
    records=read(teacher/'GPU_work_complete.json')['rows']
    expected={(r['stream_index'],r['start_tick'],r['end_tick'],r['object_index']):i for i,r in enumerate(local['metadata'])}
    seen=set();missing=0;skipped=[];coverage=[];allrows=[];durations=[]
    arrays={k:[]for k in('context','command','target','start_points','end_points','point_alive','relative_target','relative_available')}
    for job,record in zip(jobs,records):
        data=d.LoadedStream(job);seed=data.seed_masks()
        if seed is None:
            skipped.append(dict(stream_index=job['index'],case_id=job['case_id'],branch=job['branch'],reason='earliest_identity_unknown'))
            continue
        masks=load(record['masks_file']);assert sha(record['masks_file'])==record['masks_sha256'];checks+=1
        offset=len(data.history['queries'])-1
        stream=LocalFeedbackStream(data.history['cam0'][0],seed,job['instruction'],robots=bots,cfg=cfg,camera=cam)
        stream.start(data.history['cam0'][-1],masks['masks'][offset],masks['presence'][offset],
                     data.history['states'][-1],tick=job['selected_origin_tick'],elapsed_s=0.)
        segments={}
        for segment in job['stream']['segments']:
            segments[segment['kind']]=(segment,load(segment['actuation']),load(segment['packet']))
        count=0;previous=job['selected_origin_tick']
        for slot,(images,meta)in enumerate(data.native(),1):
            frame=job['stream']['frames'][slot];segment,trace,packet=segments[frame['kind']]
            first=previous-segment['absolute_start_tick'];last=meta['global_tick']-segment['absolute_start_tick']
            window={k:trace[k][first:last].copy()for k in('arm_position_target','arm_velocity_target','gripper_logical_command','dt_s')}
            began=time.monotonic()
            rows,absent=stream.arrive(images[0],masks['masks'][offset+slot],masks['presence'][offset+slot],
                window,packet['command_state'][frame['index']],
                tick=meta['global_tick'],elapsed_s=frame['elapsed_selected_s'],segment=frame['kind'],
                segment_start_tick=segment['absolute_start_tick'])
            durations.append(time.monotonic()-began)
            missing+=len(absent)
            for row in rows:
                key=(job['index'],row['start_tick'],row['end_tick'],row['object_index'])
                assert key in expected and key not in seen;seen.add(key);i=expected[key];checks+=1
                assert np.array_equal(row['context'],x[i],equal_nan=True),(key,'context',np.max(np.abs(row['context']-x[i])));checks+=1
                for name in arrays:
                    got=row[name];wanted=x[i]if name=='context'else a[name][i]
                    assert np.array_equal(got,wanted,equal_nan=True),(key,name)
                    arrays[name].append(got);checks+=1
                for field in('dt_s','segment','initial_points','matched_points'):
                    assert row[field]==local['metadata'][i][field],(key,field);checks+=1
                allrows.append(dict(source_row=i,stream_index=job['index'],start_tick=row['start_tick'],end_tick=row['end_tick'],object_index=row['object_index']))
                count+=1
            previous=meta['global_tick']
        result=stream.finish()
        assert result['rows']==count;checks+=1
        coverage.append(dict(stream_index=job['index'],case_id=job['case_id'],branch=job['branch'],rows=count,missing=result['missing']))
        print('ARRIVED_ONLY_LOCAL_STREAM_PARITY',job['index'],count,flush=True)
    assert seen==set(expected)and len(seen)==4587;checks+=1
    assert missing==len(local['missing'])==1122;checks+=1
    assert len(skipped)==4 and len(coverage)==48;checks+=1
    O.mkdir(parents=True,exist_ok=True);f=O/'stream_rows.npz'
    with f.open('xb')as out:np.savez_compressed(out,**{k:np.stack(v)for k,v in arrays.items()})
    report=dict(passed=True,checks=checks,rows=len(seen),missing=missing,unknown_streams=skipped,coverage=coverage,
        metadata=allrows,stream_arrays_sha256=sha(f),context53_command197_targets_points_relative_bitwise=True,
        no_environment_affinity_OpenCV_threads_or_numpy_RNG_import_mutation=True,
        CPU_frame_compute_seconds=dict(total=float(np.sum(durations)),mean=float(np.mean(durations)),p95=float(np.quantile(durations,.95)),maximum=float(np.max(durations))),
        source_sha256={str(p):sha(p)for p in(C/'stream_parity.py',C/'local_feedback.py',C/'primitives.py')},
        original_pure_function_source_sha256=SOURCE_SHA256,
        input_sha256={str(p):sha(p)for p in(
            T/'outputs/native_arrived_feedback_v1/local/protocol.json',
            T/'outputs/native_arrived_feedback_v1/local/data.npz',
            T/'outputs/native_local_response_v2/features.npz',
            teacher/'GPU_work_complete.json')},
        scope='Full saved52 developer streams, CPU local adapter receives one actual frame/current teacher mask at a time. External teacher is offline; no live SAM2/VLA/physics/thirdquery/runtime latency proof.',
        original_requested_cases=32,missing_cases=local['missing_cases'],online_updates=0,TTA_utility_proven=False)
    (O/'stream_parity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('FULL52_ARRIVED_LOCAL_FEEDBACK_BITWISE_PASS',checks,len(seen),missing,flush=True)
if __name__=='__main__':run()
