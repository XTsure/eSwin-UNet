from __future__ import division

import torch
from torch.nn import _reduction as _Reduction
from torch.nn import grad  # noqa: F401
from torch._overrides import has_torch_function, handle_torch_function


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    if not torch.jit.is_scripting():
        tens_ops = (input, target)
        if any([type(t) is not torch.Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                cross_entropy, tens_ops, input, target, weight=weight,
                size_average=size_average, ignore_index=ignore_index, reduce=reduce,
                reduction=reduction)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(input, 1), target, weight, None, ignore_index,
                                        None, reduction)
