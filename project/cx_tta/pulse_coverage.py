"""Report all direction/scene coverage with fixed diagnostic thresholds; no selection."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import new_run,write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    index=json.loads((a.dataset/'index.json').read_text());labels=json.loads((a.dataset/'labels_offline.json').read_text())
    if digest(a.dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Changed labels')
    groups=defaultdict(list)
    for row,label in zip(index['records'],labels):
        if row['branch']=='base':continue
        valid=label['full_horizon_valid_offline_only']
        for family,entities in label['counterfactual'].items():
            for entity,value in entities.items():
                visible=all(label[k][entity]['visible_in_any_camera'] for k in ('visibility_before','visibility_after','base_visibility_after'))
                kinds=[('translation_m',value['translation_world_m'],.005),('rotation_rad',value['rotation_world_rad'],.02)] if family=='rigid' else [('joint_fraction',np.asarray(value['delta_range_fraction'])[value['range_valid_mask']],.005)]
                for kind,y,threshold in kinds:
                    if not len(y):continue
                    groups[row['task'],entity,kind,row['horizon']].append(dict(group=row['group'],source_index=row['source_index'],branch=row['branch'],valid=valid,visible=visible,norm=float(np.linalg.norm(y)),event=bool(np.linalg.norm(y)>threshold)))
    reports=[]
    for (task,entity,kind,h),rows in sorted(groups.items()):
        valid=[r for r in rows if r['valid']];events=[r for r in valid if r['event']]
        reports.append(dict(task=task,entity=entity,kind=kind,horizon=h,rows=len(rows),full_horizon_rows=len(valid),visible_full_horizon_rows=sum(r['visible'] for r in valid),event_rows=len(events),event_scenes=len({r['group'] for r in events}),event_points=len({r['source_index'] for r in events}),total_scenes=len({r['group'] for r in rows}),total_points=len({r['source_index'] for r in rows}),max_norm=max((r['norm'] for r in valid),default=0),median_norm=float(np.median([r['norm'] for r in valid])) if valid else None))
    run=new_run('pulse_coverage',group=('reports','audits'));write_json(run/'result.json',dict(status='complete',dataset=str(a.dataset),dataset_index_sha256=digest(a.dataset/'index.json'),thresholds=dict(translation_m=.005,rotation_rad=.02,joint_fraction=.005),reports=reports,scope='Fixed-threshold offline coverage only, all scenes/directions retained; not success improvement. Horizons begin at the common pulse start; incomplete horizons separately counted.'))
    print('REPORT '+str(run/'result.json'))
    for r in reports:
        if r['horizon']==24 and r['kind'] in ('translation_m','joint_fraction') and r['event_rows']:print(json.dumps(r))

if __name__=='__main__':main()
