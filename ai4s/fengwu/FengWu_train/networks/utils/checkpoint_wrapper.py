"""
Simple replacement for fairscale's checkpoint_wrapper using torch's built-in checkpoint.
Drop-in replacement: checkpoint_wrapper(module, offload_to_cpu=True/False)
"""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class checkpoint_wrapper(nn.Module):
    def __init__(self, module: nn.Module, offload_to_cpu: bool = False):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)
