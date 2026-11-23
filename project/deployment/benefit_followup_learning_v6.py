"""Predeclared TRAIN-only cohort CV, one fresh DEV check, no TEST reads."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import benefit_followup_protocol_v2 as p
from benefit_stage_gate import CLASSES
from benefit_followup_gate_v2 import features, predict
from benefit_growth_learning import packet_at
from benefit_followup_stream_v3 import atomic

OUT = p.HOME / 'learning_v6'
DIMS = {'compact50': 50, 'dynamics33': 33, 'latent_event72': 72}


def scores(model, x):
    z = (np.asarray(x, np.float64) - model['mean']) / model['scale']
    if model['predictor_kind'] == 'outcome_softmax':
        logits = z @ np.asarray(model['weight'], np.float64) + np.asarray(model['bias'], np.float64)
        q = np.exp(logits - logits.max()); q /= q.sum(); q[2] *= .5; q /= q.sum()
    else:
        tau = float(z @ np.asarray(model['effect_weight'], np.float64) + model['effect_bias'])
        s = float(np.exp(-np.logaddexp(0., -tau)))
        q = np.array([0., s, (1-s)/2, 0.], np.float64); q /= q.sum()
    return dict(zip(CLASSES, map(float, q)))


def admitted(model, row):
    if row['features'] is None:
        return False
    q = scores(model, row['features'][model['feature_family']])
    return q['rescue'] - 2*q['harm'] > 0


def evaluate(model, rows):
    choices = [admitted(model, r) for r in rows]
    return dict(scenes=len(rows), baseline=sum(r['a'] for r in rows),
                ungated=sum(r['b'] for r in rows),
                gate=sum(r['b'] if take else r['a'] for r, take in zip(rows, choices)),
                rescues=sum(int(take and r['label'] == 'rescue') for r, take in zip(rows, choices)),
                harms=sum(int(take and r['label'] == 'harm') for r, take in zip(rows, choices)),
                admissions=sum(map(int, choices)))


def fit(rows, spec):
    family, reg, penalty = spec['feature_family'], spec['regularization'], spec['penalty']
    if spec['predictor_kind'] == 'paired_success':
        from benefit_gate_paired_probe_v5 import fit as original_fit
    else:
        from benefit_gate_latent_probe_v4 import fit as original_fit
    model = original_fit(rows, family, reg, penalty)
    model.update(schema='followup_candidate_gate_v2', predictor_kind=spec['predictor_kind'],
                 feature_source_sha256=p.sha(p.D/'benefit_followup_gate_v2.py'),
                 fitting_specification=spec, score_semantics='risk scores, not calibrated probabilities')
    return model


def load(root, split, group):
    audit = json.loads((root/'collection_audit.json').read_text())
    assert audit['passed'] and audit['complete'], ('Incomplete or failed full audit', root)
    plan = json.loads((root/'collect_plan.json').read_text()); rows = []
    for case in audit['cases']:
        if case['split'] != split:
            continue
        value = packet_at(plan, case)
        a, b = int(case['successes']['A0']), int(case['successes']['B'])
        label = ('both_success' if a else 'both_fail') if a == b else ('rescue' if b else 'harm')
        if value is None:
            assert a == b
            ff = None
        else:
            assert label == case['benefit_label']
            ff = {family: features(*value, family) for family in DIMS}
            assert all(x.shape == (DIMS[k],) and np.isfinite(x).all() for k, x in ff.items())
        rows.append(dict(a=a, b=b, label=label, features=ff, task=case['task'],
                         seed=case['seed'], group=group))
    return rows


def preregistration():
    file = OUT/'preregistration.json'; v = json.loads(file.read_text())
    assert v['candidate_count'] == len(v['specifications']) == 30
    assert v['experiment_sha256'] == p.sha(p.HOME/'experiment.json')
    assert v['namespace_amendment_sha256'] == p.sha(p.HOME/'namespace_amendment.json')
    assert all(p.sha(p.D/name) == sha for name, sha in v['source_sha256'].items())
    assert all(p.sha(p.D/name) == sha for name, sha in v['collection_source_sha256'].items())
    assert p.sha(p.CANDIDATE) == json.loads((p.HOME/'experiment.json').read_text())['adapter_sha256']
    return v


def isolation(dev_allowed=False):
    forbidden = [p.PRIOR/'test0', p.PRIOR/'final.json', p.PRIOR/'TEST_outcome_diagnosis.json',
                 p.HOME/'test0', p.HOME/'final.json']
    if not dev_allowed:
        forbidden.append(p.HOME/'dev0')
    forbidden = [q.resolve() for q in forbidden]
    def access(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path == q or q in path.parents for q in forbidden):
            raise RuntimeError('Forbidden held-out file read: '+str(path))
    sys.addaudithook(access)
    return access


def proof():
    # Numerical/interface tests only. These do not demonstrate physical benefit.
    rng = np.random.default_rng(731941)
    shapes = {'visual': (1,3,2,2,32), 'text': (1,2,32), 'state': (1,14),
              'past_visual': (1,3,2,2,32), 'past_text': (1,2,32), 'past_state': (1,14),
              'has_history': (1,1), 'ee_pose': (1,2,9), 'past_ee_pose': (1,2,9),
              'base': (1,32,14), 'candidate_delta': (1,32,14)}
    packet = {k: rng.normal(0,.1,s).astype(np.float32) for k,s in shapes.items()}
    packet['candidate_delta'][:] = np.clip(packet['candidate_delta'], -.1, .1)
    packet['candidate_delta'][:,:,[6,13]] = 0; packet['has_history'][:] = 1
    context = np.zeros(8, np.float32); before = {k:v.tobytes() for k,v in packet.items()}
    torch_rng = torch.random.get_rng_state().clone(); checked = 0
    for family, dim in DIMS.items():
        x = features(packet, context, family)
        for kind in ('outcome_softmax', 'paired_success'):
            model = dict(schema='followup_candidate_gate_v2', feature_family=family,
                         predictor_kind=kind, mean=np.zeros(dim).tolist(), scale=np.ones(dim).tolist())
            if kind == 'outcome_softmax':
                model.update(weight=rng.normal(0,.01,(dim,4)).tolist(), bias=[0.,.1,-.1,0.])
            else:
                model.update(effect_weight=rng.normal(0,.01,dim).tolist(), effect_bias=.1)
            assert scores(model,x) == predict(model,packet,context)
            assert not admitted(model,dict(features=None))
            checked += 1
    rows = []
    for v,a,b in [(-1.,1,0),(-.5,1,0),(.5,0,1),(1.,0,1)]:
        x=np.zeros(50);x[0]=v
        rows.append(dict(a=a,b=b,label='harm' if a else 'rescue',features={'compact50':x}))
    spec=dict(predictor_kind='paired_success',feature_family='compact50',regularization=1.,penalty=4.)
    m=fit(rows,spec);assert evaluate(m,rows)['gate']==4
    swapped=[dict(r,a=r['b'],b=r['a']) for r in rows];s=fit(swapped,spec)
    assert np.allclose(m['effect_weight'],-np.asarray(s['effect_weight']),atol=1e-8,rtol=0)
    neutral=[dict(r,b=r['a']) for r in rows];n=fit(neutral,spec)
    assert np.max(np.abs(n['effect_weight']))<1e-10 and abs(n['effect_bias'])<1e-10
    assert before=={k:v.tobytes() for k,v in packet.items()}
    assert torch.equal(torch_rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized()
    isolation()
    for root in (p.HOME/'dev0',p.HOME/'test0',p.PRIOR/'test0'):
        try: (root/'never_read.json').open()
        except RuntimeError as e: assert 'Forbidden held-out file read' in str(e)
        else: raise AssertionError('Held-out access was not denied')
    result=dict(passed=True,cpu_only=True,cuda_initialized=False,exact_deployed_predictor_comparisons=checked,
                labels_swapped_effect_reversed=True,neutral_effect_zero=True,synthetic_benefit_not_research_evidence=True,
                packet_and_Torch_RNG_unchanged=True,actual_DEV_and_TEST_open_calls_denied=True,
                learner_sha256=p.sha(__file__),preregistration_sha256=p.sha(OUT/'preregistration.json'))
    atomic(OUT/'cpu_proof.json',result);print('FOLLOWUP_LEARNING_CPU_PROOF',json.dumps(result),flush=True)


def train():
    v=preregistration(); isolation()
    if (OUT/'TRAIN_selected_candidate.json').exists():
        raise RuntimeError('TRAIN candidate already selected; preserve it')
    cert=json.loads((OUT/'cpu_proof.json').read_text())
    assert cert['passed'] and cert['learner_sha256']==p.sha(__file__)
    assert cert['preregistration_sha256']==p.sha(OUT/'preregistration.json')
    exp=json.loads((p.HOME/'experiment.json').read_text());groups=[];inputs={};seen=set()
    def add(root,split,name,count):
        audit=root/'collection_audit.json';rows=load(root,split,name);assert len(rows)==count
        for row in rows:
            key=(row['task'],row['seed']);assert key not in seen;seen.add(key)
        groups.append(rows);inputs[str(audit)]=p.sha(audit)
    for i,item in enumerate(exp['base_training_sources']):
        root=p.BASE if i==0 else p.PRIOR/('train0','train1','dev0')[i-1]
        assert p.sha(root/'collection_audit.json')==item['audit_sha256']
        add(root,item['original_split'],'base'+str(i),item['scenes'])
    complete=[]
    for block in exp['new_TRAIN_block_order']:
        file=p.HOME/block/'collection_audit.json'
        if not file.exists():break
        add(file.parent,'TRAIN',block,40);complete.append(block)
    assert complete,'Need at least one fully audited prospective cohort'
    rows=sum(groups,[]);counts={k:sum(r['label']==k for r in rows) for k in CLASSES}
    assert counts['rescue']>=12 and counts['harm']>=12,('Complete next prescribed TRAIN cohort before fitting',counts,complete)
    grid=[]
    for index,spec in enumerate(v['specifications']):
        folds=[evaluate(fit(sum([g for i,g in enumerate(groups) if i!=k],[]),spec),held) for k,held in enumerate(groups)]
        cv={key:sum(f[key] for f in folds) for key in folds[0]}
        grid.append(dict(specification=spec,folds=folds,CV=cv))
        print('TRAIN_CV',index,cv,flush=True)
    best=max(range(len(grid)),key=lambda i:(grid[i]['CV']['gate'],-grid[i]['CV']['harms'],
             -grid[i]['CV']['admissions'],-grid[i]['specification']['dimension'],
             grid[i]['specification']['regularization'],-grid[i]['specification']['penalty'],-i))
    selected=grid[best];model=fit(rows,selected['specification'])
    model.update(train_scenes=len(rows),train_class_counts=counts,validation_used_for_fit=False,
                 DEV_used_for_selection=False,TEST_loaded=False,training_source_sha256=p.sha(__file__),
                 preregistration_sha256=p.sha(OUT/'preregistration.json'),data_audit_sha256=inputs)
    atomic(OUT/'TRAIN_selection.json',dict(passed=True,groups=[len(g) for g in groups],completed_new_TRAIN=complete,
           train_counts=counts,grid=grid,selected=selected,DEV_loaded=False,TEST_loaded=False,learner_sha256=p.sha(__file__)))
    atomic(OUT/'TRAIN_selected_candidate.json',model)
    assert not torch.cuda.is_initialized();print('TRAIN_SELECTED',json.dumps(selected),flush=True)


def dev():
    preregistration();file=OUT/'TRAIN_selected_candidate.json';model=json.loads(file.read_text())
    assert model['training_source_sha256']==p.sha(__file__)
    assert model['preregistration_sha256']==p.sha(OUT/'preregistration.json')
    if (OUT/'DEV_evaluation_started.json').exists():
        raise RuntimeError('DEV evaluation already started; do not repeat or tune')
    atomic(OUT/'DEV_evaluation_started.json',dict(candidate_sha256=p.sha(file),started_unix=time.time()))
    isolation(dev_allowed=True);rows=load(p.HOME/'dev0','DEV','new_DEV');assert len(rows)==40
    result=evaluate(model,rows);ready=result['gate']>max(result['baseline'],result['ungated']) and result['rescues']>0
    report=dict(passed=True,DEV=result,ready_for_independent_TEST=ready,DEV_checked_once=True,
                TEST_loaded=False,candidate_sha256=p.sha(file),learner_sha256=p.sha(__file__),
                DEV_audit_sha256=p.sha(p.HOME/'dev0/collection_audit.json'))
    atomic(OUT/'DEV_result.json',report)
    if ready:
        assert not (p.HOME/'gate.json').exists(),'Do not overwrite a frozen gate'
        model.update(DEV_check_sha256=p.sha(OUT/'DEV_result.json'),frozen_unix=time.time())
        atomic(p.HOME/'gate.json',model)
    assert not torch.cuda.is_initialized();print('FOLLOWUP_DEV_RESULT',json.dumps(report),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=('proof','train','dev'),default='proof');args=a.parse_args()
    torch.set_num_threads(1);assert not torch.cuda.is_initialized();OUT.mkdir(exist_ok=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16]);preregistration()
    {'proof':proof,'train':train,'dev':dev}[args.phase]()

if __name__=='__main__':main()
