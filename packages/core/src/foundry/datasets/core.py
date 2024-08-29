from typing import Generic, TypeVar, Mapping, Callable

from foundry.core.dataclasses import dataclass, field
from foundry.data import Data
from foundry.data.transform import Transform
from foundry.data.normalizer import Normalizer

import logging
logger = logging.getLogger("foundry.datasets")

T = TypeVar('T')

@dataclass
class Dataset(Generic[T]):
    splits: Mapping[str, Data[T]]
    normalizers: Mapping[str, Callable[[], Normalizer[T]]] = field(default_factory=dict)
    transforms: Mapping[str, Callable[[], Transform[T]]] = field(default_factory=dict)

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
