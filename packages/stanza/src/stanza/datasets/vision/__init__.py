from stanza.datasets import DatasetRegistry, Dataset
from stanza.dataclasses import dataclass
from stanza.util.registry import from_module

from typing import Tuple

import jax

@dataclass
class ImageDataset(Dataset[jax.Array]):
    pass

@dataclass
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
            },
            transforms={}
        )

image_class_datasets : DatasetRegistry[ImageClassDataset] = DatasetRegistry[Dataset]()
"""Datasets containing (image, label) pairs,
where label is one-hot encoded."""
image_class_datasets.extend("mnist", from_module(".mnist", "datasets"))
image_class_datasets.extend("cifar", from_module(".cifar", "datasets"))