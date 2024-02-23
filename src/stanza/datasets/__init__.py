from typing import Callable, Generic, TypeVar, Optional, Union, Protocol, Iterable, Tuple

from stanza import struct
from stanza.data import Data
from stanza.util.registry import Registry, register_module

import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@struct.dataclass
class Dataset(Generic[T]):
    splits: dict[str, Data[T]] = struct.field(default_factory=dict)

DatasetRegistry = Registry

image_label_datasets : DatasetRegistry[Dataset] = DatasetRegistry[Dataset]()
"""Datasets containing (image, label) pairs,
where label is one-hot encoded.
"""

image_label_datasets.defer(register_module(".mnist", "dataset_registry"))

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
