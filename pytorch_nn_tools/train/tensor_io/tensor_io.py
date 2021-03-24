import os
from abc import ABCMeta, abstractmethod
from typing import Union, BinaryIO, IO


class TensorIO(metaclass=ABCMeta):
    """
    Interface for low level saving and loading tensors (models)
    """
    @abstractmethod
    def save(self, obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
        pass

    @abstractmethod
    def load(self, f):
        pass


