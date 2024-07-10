from stanza.datasets import DatasetRegistry, Dataset
from stanza.dataclasses import dataclass
from stanza.util.registry import from_module

import jax

from typing import TypeVar, Generic

T = TypeVar('T')

@dataclass
class EnvDataset(Dataset[T], Generic[T]):
    def create_env(self):
        raise NotImplementedError()

datasets : DatasetRegistry[EnvDataset] = DatasetRegistry()
datasets.extend("pusht", from_module(".pusht", "datasets"))
datasets.extend("robomimic", from_module(".robomimic", "datasets"))