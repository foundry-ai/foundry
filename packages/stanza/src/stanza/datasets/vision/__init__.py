from stanza.datasets import DatasetRegistry, Dataset
from stanza.dataclasses import dataclass, field
from stanza.util.registry import from_module

from typing import Tuple, Sequence

import jax

@dataclass
class Image:
    pixels: jax.Array

@dataclass
class LabeledImage(Image):
    label: jax.Array

@dataclass
class ImageDataset(Dataset[Image]):
    pass

@dataclass
class ImageClassDataset(Dataset[LabeledImage]):
    classes: Sequence[str] = field(default_factory=tuple)

image_class_datasets : DatasetRegistry[ImageClassDataset] = DatasetRegistry[Dataset]()
"""Datasets containing (image, label) pairs,
where label is one-hot encoded."""
image_class_datasets.extend("mnist", from_module(".mnist", "datasets"))
image_class_datasets.extend("cifar", from_module(".cifar", "datasets"))

image_datasets : DatasetRegistry[ImageDataset] = DatasetRegistry[Dataset]()
image_datasets.extend("mnist", from_module(".mnist", "datasets"))
image_datasets.extend("cifar", from_module(".cifar", "datasets"))
image_datasets.extend("celeb_a", from_module(".celeb_a", "datasets"))