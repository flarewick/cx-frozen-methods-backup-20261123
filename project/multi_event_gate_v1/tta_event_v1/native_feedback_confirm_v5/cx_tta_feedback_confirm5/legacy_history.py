"""Actual observed previousbase-only interval; no invented physical duration."""
import numpy as np
from .paths import read,write
from .contract import load_packet

def encode(rows):
    return dict(queries=np.arange(len(rows),dtype=np.int64),states=np.stack([r['command_state'] for r in rows]),
        base_chunks=np.stack([r['base'] for r in rows]),**{'cam'+str(i):np.stack([r['rgb'][i] for r in rows]) for i in range(3)})
def save_history(folder,rows):
    if not rows:raise ValueError('No original observed query history')
    with (folder/'observed_history.npz').open('xb') as f:np.savez_compressed(f,**encode(rows))
    write(folder/'observed_history.json',dict(input_images=[r['images'] for r in rows],physical_interval_duration_available=False,
        semantics='Each query observation already passed to frozen policy; previousbase_chunks beforefork actuallyexecuted. Lastbase is currentonceproposal, not pastresponse. No extra render/observation/inference.'))
def read_history(folder):return load_packet(folder/'observed_history.npz')
def verify_history(folder):
    from cx_tta.eval_diagnostics import array_hash
    h=read_history(folder);n=len(h['queries']);discovery=read(folder/'discovery.json');meta=read(folder/'observed_history.json')
    if set(h)!={'queries','states','base_chunks','cam0','cam1','cam2'} or n!=discovery['sampler_calls'] or not np.array_equal(h['queries'],np.arange(n)):
        raise ValueError('Originalquery history alignment changed')
    proposal=load_packet(folder/'proposal.npz');initial=load_packet(folder/'initial.npz');rgb=load_packet(folder/'Z_reference_rgb.npz')
    if h['states'].shape!=(n,14) or h['base_chunks'].shape!=(n,32,14) or n!=len(proposal['prefix'])+1:raise ValueError('History temporal shape mismatch')
    if not np.array_equal(h['base_chunks'][:-1],proposal['prefix']) or not np.array_equal(h['base_chunks'][-1],proposal['base']):raise ValueError('History commands not actualprefix/currentproposal')
    if not np.array_equal(h['states'][-1],initial['command_state']):raise ValueError('History not ending at currentobserved fork')
    for c in range(3):
        images=h['cam'+str(c)]
        if images.dtype!=np.uint8 or len(images)!=n or not np.array_equal(images[-1],rgb['initial_'+str(c)]):raise ValueError('History RGB end differs from encoderfork')
        for image,row in zip(images,meta['input_images']):
            if array_hash(image)!=row[str(c)]:raise ValueError('History RGB differs from recorded encoder input')
    if meta['physical_interval_duration_available']:raise ValueError('No fabricated physicaltime for unlogged pastprefix')
def cpu_checks():
    row=dict(rgb=tuple(np.zeros((12+i,14+i,3),np.uint8) for i in range(3)),command_state=np.zeros(14,np.float32),base=np.zeros((32,14),np.float32))
    h=encode([row,row]);assert h['base_chunks'].shape==(2,32,14) and h['cam2'].shape==(2,14,16,3)
    h['cam0'][0,0,0,0]=1;assert not row['rgb'][0].any()
    return dict(causal_history_copies_original_query_arrays=True,history_supports_native_different_camera_shapes=True)
