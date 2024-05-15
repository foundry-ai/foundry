from typing import Generic, TypeVar, Mapping, Tuple, Callable

from stanza import struct
from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer
from stanza.util.registry import Registry, from_module, transform_result

import jax
import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@struct.dataclass
class Dataset(Generic[T]):
    splits: Mapping[str, Data[T]]
    normalizers: Mapping[str, Callable[[], Normalizer[T]]]
    transforms: Mapping[str, Callable[[], Transform[T]]]

DatasetRegistry = Registry

@struct.dataclass
class ImageDataset(Dataset[jax.Array]):
    pass

@struct.dataclass
class ImageClassDataset(Dataset[Tuple[jax.Array, jax.Array]]):
    classes: list[str]

    def as_image_dataset(self) -> ImageDataset:
        def map_normalizer(normalizer_builder):
            def mapped(*args, **kwargs):
                return normalizer_builder(*args, **kwargs).map(lambda x: x[0])
            return mapped

        return ImageDataset(
            splits={k: v.map(lambda x: x[0]) for k, v in self.splits.items()},
            normalizers={
                k: map_normalizer(v)
                for k, v in self.normalizers.items()
            }
        )

image_class_datasets : DatasetRegistry[ImageClassDataset] = DatasetRegistry[Dataset]()
"""Datasets containing (image, label) pairs,
where label is one-hot encoded.
"""
image_class_datasets.defer(from_module(".mnist", "registry"))
image_class_datasets.defer(from_module(".cifar", "registry"))

image_datasets : DatasetRegistry[ImageDataset] = DatasetRegistry[Dataset]()
image_datasets.defer(image_class_datasets, transform_result(lambda x: x.as_image_dataset()))
image_datasets.defer(from_module(".celeb_a", "registry"))

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
