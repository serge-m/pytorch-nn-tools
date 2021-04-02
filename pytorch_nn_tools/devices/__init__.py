import collections
from collections.abc import Iterable, Collection
from typing import Union

import torch
from pytorch_nn_tools.convert.sized_map import sized_map


def apply_recursively(func, x):
    """
    Apply `func` recursively to `x`.

    >>> result = apply_recursively(\
            lambda val: val+1, {'a': 10, 'b': [20,30,{'c':40}]})
    >>> assert result == {'a': 11, 'b': [21,31,{'c':41}]}
    >>> from collections import OrderedDict
    >>> result = apply_recursively(\
            lambda val: val+1,\
            OrderedDict([ \
                ('a', 10),  \
                ('b', (20,30)) \
            ]) \
        )
    >>> assert result == OrderedDict([('a', 11), ('b', (21,31))])
    >>> assert type(result) == OrderedDict
    >>> assert type(result['b']) == tuple
    """
    if isinstance(x, collections.abc.Sequence):
        return type(x)([apply_recursively(func, i) for i in x])
    if isinstance(x, collections.abc.Mapping):
        return type(x)(
            (k, apply_recursively(func, v))
            for k, v in x.items()
        )
    return func(x)


def to_device(data, device):
    """
    Sends the input to a given device.

    If the input is a Tensor it uses non_blocking=True for speed up.
    Some explanation can be found here https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/7
    """

    def _inner(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        if isinstance(x, torch.nn.Module):
            return x.to(device)
        elif hasattr(x, "to_device"):
            return x.to_device(device)
        else:
            return x

    return apply_recursively(_inner, data)


def iter_to_device(iterable: Union[Iterable, Collection], device) -> Union[Iterable, Collection]:
    """
    Sends each element of the iterable to a device during iteration.
    The __len__ of the iterable is preserved if available.
    @param iterable: iterable to process
    @param device: e.g. 'cpu' or torch.device('cuda')
    @return:

    >>> tensors = [torch.tensor([1]), torch.tensor([2])]
    >>> list(iter_to_device(tensors, 'cpu'))
    [tensor([1]), tensor([2])]
    >>> len(iter_to_device(tensors, 'cpu'))
    2
    """
    return sized_map(lambda b: to_device(b, device), iterable)
