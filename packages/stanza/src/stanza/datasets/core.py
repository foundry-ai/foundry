from typing import Generic, TypeVar, Mapping, Callable

from stanza.dataclasses import dataclass
from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer

import jax
import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@dataclass
class Dataset(Generic[T]):
    splits: Mapping[str, Data[T]]
    normalizers: Mapping[str, Callable[[], Normalizer[T]]]
    transforms: Mapping[str, Callable[[], Transform[T]]]

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
