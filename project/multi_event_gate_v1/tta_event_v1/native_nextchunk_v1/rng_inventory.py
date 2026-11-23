"""Read frozen native RNG fingerprints only; never train/select by physics."""
from pathlib import Path
import json,hashlib
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_nextchunk_v1'
R=T/'outputs/native_stage_v1/collections/5d5369942959181f58a604c4'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    rows=[];missing=[];inputs={}
    for f in sorted((R/'sessions').rglob('case.json')):
        case=read(f)
        if case.get('role')!='development':continue
        folder=f.parent
        if (folder/'unavailable.json').exists():
            missing.append(dict(case_id=case['case_id'],unavailable=read(folder/'unavailable.json')));continue
        for branch in ('Z_reference','C_original','Z_sham'):
            p=folder/(branch+'.json')
            if not p.exists():
                # Do not silently use another spelling for the frozen branch.
                continue
            m=read(p);before=m['prefix_rng_sha256']
            hashes=[s['rng_sha256'] for s in m['samples']]
            rows.append(dict(case_id=case['case_id'],branch=branch,samples=len(hashes),
                     changed_from_prefix=sum(h!=before for h in hashes),
                     unique_hashes=len(set([before]+hashes)),prefix_rng_components=m.get('prefix_rng_components'),
                     final_equals_prefix=bool(hashes and hashes[-1]==before)))
            inputs[str(p)]=sha(p)
    assert rows
    report=dict(complete=True,source_cases=len({r['case_id'] for r in rows}),branches=len(rows),
         rows=rows,source_missing=missing,
         all_sampled_RNG_fingerprints_constant=all(r['unique_hashes']==1 for r in rows),
         all_terminal_RNG_fingerprints_equal_prefix=all(r['final_equals_prefix'] for r in rows),
         source_sha256=sha(Path(__file__)),input_sha256=inputs,
         scope='Frozen native single-origin diagnostic RNG fingerprint inventory only; no actual sampler reconstruction, no CUDA handoff/reference parity or TTA success. Fingerprints at sampled readouts do not prove intermediate draw-free execution or identify hidden streams. No model inputs/label selection.')
    O.mkdir(parents=True,exist_ok=True)
    file=O/'native_rng_inventory.json'
    assert not file.exists();file.write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_RNG_INVENTORY',report['source_cases'],len(rows),report['all_sampled_RNG_fingerprints_constant'],report['all_terminal_RNG_fingerprints_equal_prefix'])
    print('BRANCHES',sorted({r['branch'] for r in rows}))
if __name__=='__main__':run()
