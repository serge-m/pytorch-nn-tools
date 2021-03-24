import os
from typing import Union, BinaryIO, IO
import torch
import torch_xla.core.xla_model as xm

from pytorch_nn_tools.train.tensor_io.tensor_io import TensorIO


class TorchXlaTensorIO(TensorIO):
    def save(self, obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
        return xm.save(obj, f)

    def load(self, f):
        return torch.load(f)
