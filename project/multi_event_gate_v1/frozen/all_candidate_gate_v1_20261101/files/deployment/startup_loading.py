"""Skip only random fills overwritten by a strictly loaded checkpoint."""
from contextlib import contextmanager


@contextmanager
def skip_overwritten_random_init():
    import torch
    from transformers.modeling_utils import PreTrainedModel
    names = ('uniform_', 'normal_', 'trunc_normal_', 'xavier_uniform_',
             'xavier_normal_', 'kaiming_uniform_', 'kaiming_normal_')
    originals = {name: getattr(torch.nn.init, name) for name in names}

    def unchanged_tensor(tensor, *args, **kwargs):
        return tensor

    original_hf_initializer = PreTrainedModel._initialize_weights

    def initialized_without_random_fill(self, module):
        module._is_hf_initialized = True

    try:
        # Keep post_init and tie_weights intact; omit only checkpoint-overwritten
        # initialization, including HF initializers which use Tensor.uniform_.
        PreTrainedModel._initialize_weights = initialized_without_random_fill
        for name in names:
            setattr(torch.nn.init, name, unchanged_tensor)
        yield
    finally:
        PreTrainedModel._initialize_weights = original_hf_initializer
        for name, function in originals.items():
            setattr(torch.nn.init, name, function)
