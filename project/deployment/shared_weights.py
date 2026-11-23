"""Share only frozen parameters; buffers, compilers, RNG and policy state stay private."""
import hashlib
import json


def parameter_layout(model):
    groups, layout = {}, {}
    for name, value in model.named_parameters(remove_duplicate=False):
        canonical = groups.setdefault(id(value), name)
        layout[name] = dict(canonical=canonical, shape=list(value.shape),
                            dtype=str(value.dtype), stride=list(value.stride()))
    return layout


def export_weights(model):
    layout = parameter_layout(model)
    params = dict(model.named_parameters(remove_duplicate=False))
    tensors = {row['canonical']: params[row['canonical']].detach()
               for row in layout.values()}
    return dict(layout=layout, tensors=tensors)


def bind_weights(model, payload, device):
    import torch
    layout = parameter_layout(model)
    if payload['layout'] != layout:
        raise ValueError('Shared parameter names/shapes/dtypes/aliases differ')
    canonical = {v['canonical'] for v in layout.values()}
    if set(payload['tensors']) != canonical:
        raise ValueError('Missing or extra shared parameters')
    # Validate everything before mutating the model.
    for name, tensor in payload['tensors'].items():
        row=layout[name]
        if str(tensor.device)!=device or tensor.requires_grad or list(tensor.shape)!=row['shape'] or str(tensor.dtype)!=row['dtype'] or list(tensor.stride())!=row['stride']:
            raise ValueError('Invalid shared tensor: '+name)
    replacements = {k:torch.nn.Parameter(v, requires_grad=False) for k,v in payload['tensors'].items()}
    for name, row in layout.items():
        parent, _, leaf = name.rpartition('.')
        module = model.get_submodule(parent) if parent else model
        module._parameters[leaf] = replacements[row['canonical']]
    if parameter_layout(model)!=layout:
        raise RuntimeError('Shared assignment changed parameter aliases')
    return dict(parameter_groups=len(canonical),parameter_names=len(layout),
                parameter_bytes=sum(v.numel()*v.element_size() for v in payload['tensors'].values()),
                layout_sha256=hashlib.sha256(json.dumps(layout,sort_keys=True).encode()).hexdigest(),
                buffers_shared=False)


def versions(model):
    return {name:(id(p),p.data_ptr(),p._version) for name,p in model.named_parameters()}
