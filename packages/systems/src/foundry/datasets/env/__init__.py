from foundry.datasets.core import DatasetRegistry, Dataset
from foundry.core.dataclasses import dataclass
from foundry.data.sequence import SequenceData
from typing import TypeVar, Generic

T = TypeVar('T')

@dataclass
class EnvDataset(Dataset[T], Generic[T]):
    def split(name) -> SequenceData[T, None]:
        return None

    def create_env(self):
        raise NotImplementedError()

def register_all(registry: DatasetRegistry[EnvDataset], prefix=None):
    from . import pusht, robomimic
    pusht.register(registry, prefix=prefix)
    robomimic.register_all(registry, prefix=prefix)