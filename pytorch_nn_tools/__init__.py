"""Top-level package for pytorch-nn-tools."""

__author__ = """SergeM"""
__email__ = 'serge-m@users.noreply.github.com'
__version__ = '0.1.0'

from .hook import Hook, _hook_inner
from .structure import get_output_sizes, LayerSize, NameGetter
from .memory import dummy_batch
