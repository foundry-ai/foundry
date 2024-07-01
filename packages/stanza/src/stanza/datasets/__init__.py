from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer
from typing import Any
from stanza.util.registry import Registry, from_module

from .core import (
    Dataset
)

DatasetRegistry = Registry

datasets : DatasetRegistry[Dataset] = DatasetRegistry[Dataset]()
datasets.extend("vision", from_module(".vision", "datasets"))
datasets.extend("env", from_module(".env", "datasets"))
datasets.extend("nlp", from_module(".nlp", "datasets"))

def load(path: str, /, **kwargs : dict[str, Any]):
    return datasets.create(path, **kwargs)
