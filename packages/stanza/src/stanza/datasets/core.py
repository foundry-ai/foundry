from typing import Generic, TypeVar, Mapping, Callable

from stanza.dataclasses import dataclass, field
from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer

import logging
logger = logging.getLogger("stanza.datasets")

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
