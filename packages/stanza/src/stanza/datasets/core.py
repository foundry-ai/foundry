from typing import Generic, TypeVar, Mapping, Tuple, Callable, Any

from stanza.dataclasses import dataclass
from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer
from stanza.util.registry import Registry, from_module

import jax
import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@dataclass
class Dataset(Generic[T]):
    splits: Mapping[str, Data[T]]
    normalizers: Mapping[str, Callable[[], Normalizer[T]]]
    transforms: Mapping[str, Callable[[], Transform[T]]]

DatasetRegistry = Registry

datasets : DatasetRegistry[Dataset] = DatasetRegistry[Dataset]()
datasets.extend("vision", from_module(".vision", "datasets"))
datasets.extend("env", from_module(".env", "datasets"))
datasets.extend("nlp", from_module(".nlp", "datasets"))

def load(path: str, /, **kwargs : dict[str, Any]):
    return datasets.create(path, **kwargs)

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
