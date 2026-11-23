"""Offline baseline contact snapshots relative to the fixed pulse; never online routing."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter,defaultdict
from .paths import ROOT,new_run,write_json
from .audit import digest
from .robot_kinematics import RobotFK
from .cartesian_pulse_batch import DEFAULT_PLAN,validate_plan


def main():
    plan=json.loads(DEFAULT_PLAN.read_text());validate_plan(plan);records=[]
    for task in sorted({j['task'] for j in plan['jobs']}):
        index=json.loads((ROOT/'outputs/cartesian_pulse_batches'/plan['batch_id']/task/'index.json').read_text())
        if index['status']!='complete' or index['plan_sha256']!=digest(DEFAULT_PLAN):raise ValueError('Changed batch')
        for job in index['records']:
            path=Path(job['run'])
            if digest(path/'result.json')!=job['result_sha256']:raise ValueError('Changed result')
            result=json.loads((path/'result.json').read_text());config=json.loads((Path(job['source'])/'resolved_config.json').read_text())
            robot_names=set()
            for side in ('left','right'):
                robot_names.update(link.get('name') for link in ET.parse(RobotFK(config,side).urdf).getroot().findall('link'))
            branch=result['branches']['base'];ids={i:entity for entity,items in branch['entity_id_map_offline'].items() for i in items};contacts={}
            for frame in branch['frames']:
                names=set()
                for pair in frame['contacts_offline_only']['pairs']:
                    for body in (0,1):
                        if pair['entity_names'][body] in robot_names and pair['entity_ids'][1-body] in ids:
                            names.add(ids[pair['entity_ids'][1-body]])
                if names:contacts[str(frame['offset'])]=sorted(names)
            offsets=sorted(map(int,contacts));start=job['pulse_start'];end=start+8
            category=('contact_at_pulse_start' if start in offsets else 'contact_in_pulse_snapshots' if any(start<o<=end for o in offsets) else 'contact_only_after_pulse' if offsets and min(offsets)>end else 'other_contact_outside_pulse' if offsets else 'no_recorded_robot_object_contact')
            records.append(dict(task=task,seed=job['seed'],job=job['id'],category=category,contact_entities_by_offset=contacts,first_recorded_offset=min(offsets) if offsets else None))
    summary={task:dict(Counter(r['category'] for r in records if r['task']==task)) for task in sorted({r['task'] for r in records})}
    run=new_run('pulse_contact_coverage',group=('reports','audits'));write_json(run/'result.json',dict(status='complete',records=records,summary=summary,scope='Baseline snapshots only; body names matched to robot URDF, object IDs matched to offline entity map. First recorded contact is not first physical contact, and absence between snapshots cannot be inferred. No branch effect selection or policy input.'))
    print(json.dumps(dict(report=str(run/'result.json'),summary=summary),indent=2))

if __name__=='__main__':main()
