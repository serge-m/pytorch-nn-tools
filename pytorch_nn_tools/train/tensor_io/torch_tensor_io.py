import os
from typing import Union, BinaryIO, IO

import torch

from pytorch_nn_tools.train.tensor_io.tensor_io import TensorIO


class TorchTensorIO(TensorIO):
    def save(self, obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
        return torch.save(obj, f)

    def load(self, f):
        return torch.load(f)
