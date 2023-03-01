from typing import TypeVar

from torch import Tensor
from torch.utils.data import TensorDataset

from .data import Data

TensorData = TypeVar("TensorData", TensorDataset, Tensor)
SomeData = TypeVar("SomeData", bound=Data)
