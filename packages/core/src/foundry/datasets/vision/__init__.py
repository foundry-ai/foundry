from foundry.datasets.core import DatasetRegistry, Dataset

from foundry.core.dataclasses import dataclass
from foundry.core.typing import Array

from typing import Sequence

@dataclass
class Image:
    pixels: Array

@dataclass
class LabeledImage(Image):
    label: Array

class ImageDataset(Dataset[Image]):
    pass

class ImageClassDataset(Dataset[LabeledImage]):
    @property
    def classes(self) -> Sequence[str]:
        raise NotImplementedError()

def register_all(registry: DatasetRegistry, prefix=None):
    from . import cifar
    # from . import celeb_a
    from . import ffhq
    # from . import imagenette
    from . import mnist
    cifar.register(registry, prefix=prefix)
    ffhq.register(registry, prefix=prefix)
    mnist.register(registry, prefix=prefix)
    # celeb_a.register(registry, prefix=prefix)
    # imagenette.register(registry, prefix=prefix)
    # mnist.register(registry, prefix=prefix)