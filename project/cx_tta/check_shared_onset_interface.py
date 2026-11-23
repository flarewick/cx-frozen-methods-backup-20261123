"""Replay all frozen pilot packets through the runtime selector; no fitting."""
import argparse
from pathlib import Path
import numpy as np
from .paths import new_run,write_json,output_path
from .native_history_cache import read,arrays,require
from .empirical_onset_gate import GATES,load_frozen_gate,EmpiricalOnsetGate,selected_actions
from .arm_prescription_policy import compose
from .onset_observables import validate_packet
from .recovery_contract import rng_capture,rng_fingerprint
from .audit import digest


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--gate-run',type=Path,required=True);a=ap.parse_args();gate_run=output_path(a.gate_run)
    p=read(gate_run/'protocol.json')['recipe'];z=arrays(Path(p['data'])/'observables.npz')
    run=new_run('shared_gate_interface',group=('reports','diagnostics'));print('RUN',run,flush=True)
    results={}
    for condition,name in GATES.items():
        model,sha=load_frozen_gate(gate_run,condition);gate=EmpiricalOnsetGate(model)
        predictions={v['index']:v for stage in ('fit','calibration') for v in read(gate_run/(stage+'_predictions.json'))[name]}
        selected=0
        for i,v in predictions.items():
            packet=validate_packet({k:x[i:i+1] for k,x in z.items()});base=packet['base'][0];raw=packet['candidate_delta'][0]
            before=rng_fingerprint(rng_capture());gate.reset();gate.step(0,[1,3]);d=gate.step(1,[1,3],packet)
            require(d['scores']==v['scores'] and d['accepted']==v['decisions']['2'],'Exact saved scores and selection')
            actions,delta,_=selected_actions(base,raw,1,[1,3],d)
            reference=compose(base,raw,'F_first_full' if d['accepted'] else 'Z_zero',1,[1,3])
            require(actions.tobytes()==reference[0].tobytes() and delta.tobytes()==reference[1].tobytes(),'Exact selected action branch')
            require(actions[:,[6,13]].tobytes()==base[:,[6,13]].tobytes(),'Exact base grippers')
            for q in (2,3,4):
                later=gate.step(q,[1,3]);act,dd,_=selected_actions(base,raw,q,[1,3],later)
                require(not later['scored'] and act.tobytes()==base.tobytes() and not dd.any(),'Default permanent exit')
            require(before==rng_fingerprint(rng_capture()),'No RNG consumption by gate')
            selected+=int(d['accepted'])
        results[condition]=dict(packets=len(predictions),accepted=selected,rejected=len(predictions)-selected,
            exact_score_replay=True,exact_actions=True,reset_and_exit=True,RNG_unchanged=True,model_sha256=sha)
    write_json(run/'result.json',dict(passed=True,models=results,training=False,GPU=False,success_rate_claim=False,
        source_sha256={str(Path(__file__).resolve()):digest(Path(__file__)),str(gate_run/'result.json'):digest(gate_run/'result.json')}))
    print('PASS',results,flush=True)

if __name__=='__main__':main()
