"""Use existing same-input GPU probe outputs to audit numerical sensitivity on CPU."""
import itertools
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .paths import ROOT,write_json
from .dataset import context_features,expert_chunk
from .deploy import load_adapter
from .evaluate_b2 import BEST
from .audit import digest


def main():
    folder=ROOT/'outputs/diagnostics/inference'
    bundle=folder/'fixed_inputs_20260926_130935_129985'
    inputs=json.loads((bundle/'inputs.json').read_text())['records']
    names={'compiled_1':'probe_20260926_131002_862049','compiled_2':'probe_20260926_132712_312205',
           'eager_1':'probe_20260926_132408_487843','eager_2':'probe_20260926_134118_336823'}
    model=load_adapter(BEST);caps=model.caps.numpy();rows=[]
    for i,rec in enumerate(inputs):
        assert digest(Path(rec['source']))==rec['source_sha256']
        with np.load(bundle/rec['path']) as z:
            context=context_features([z[f'image_{j}'] for j in range(3)],z['state'])
        with h5py.File(rec['source']) as h:
            target,valid=expert_chunk(h['joint_action/vector'][:],rec['frame'])
        modes={};outputs={}
        for mode,name in names.items():
            base=np.load(folder/name/f'r0_input{i}.npy')
            with torch.no_grad():pred,delta=model(torch.from_numpy(context)[None],torch.from_numpy(base)[None])
            pred,delta=pred[0].numpy(),delta[0].numpy();outputs[mode]=(base,pred,delta)
            mse=lambda x:float(np.mean(np.square((x[valid]-target[valid])/caps)))
            modes[mode]={'base_normalized_mse':mse(base),'b2_normalized_mse':mse(pred),
                         'correction_rms':float(np.sqrt(np.mean(delta[valid]**2)))}
        pairs={}
        for a,b in itertools.combinations(names,2):
            x,y=outputs[a],outputs[b]
            pairs[a+'__'+b]={'base_rms_difference':float(np.sqrt(np.mean((x[0][valid]-y[0][valid])**2))),
                            'base_max_abs_difference':float(np.max(np.abs(x[0][valid]-y[0][valid]))),
                            'correction_max_abs_difference':float(np.max(np.abs(x[2][valid]-y[2][valid])))}
        rows.append({'task':rec['task'],'frame':rec['frame'],'source':rec['source'],'modes':modes,'pairs':pairs})
    summary={task:{mode:{key:float(np.mean([r['modes'][mode][key] for r in rows if r['task']==task]))
                                for key in ('base_normalized_mse','b2_normalized_mse','correction_rms')}
                        for mode in names} for task in sorted({r['task'] for r in rows})}
    result={'scope':'8 development expert frames, not validation loss or success rate; same explicit noise across modes',
            'limitations':['not a recomputation of the old full cache: noise seeds differ from old cache',
                            'expert trajectory states do not cover closed-loop failure states',
                            'uses original decoded expert RGB for adapter context, as in training'],
            'checkpoint_sha256':digest(BEST),'rows':rows,'summary':summary}
    write_json(ROOT/'outputs/reports/audits/probe_adapter_sensitivity_20260926.json',result)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
