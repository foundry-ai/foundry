from typing import Callable, Generic, TypeVar

from stanza.struct import struct, field
from stanza.data import Data

import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@struct(frozen=True)
class Dataset:
    splits: dict[str, Data] = field(default_factory=dict)

class Registry(Generic[T]):
    def __init__(self):
        self._datasets = {}
        self._deffered = []
    
    def register(self, name: str, loader: Callable):
        logger.debug(f"Registered dataset: {name}")
        self._datasets[name] = loader
    
    def register_all(self, datasets: dict[str, Callable]):
        if datasets is None:
            return
        if hasattr(datasets, "datasets"):
            datasets = datasets.datasets
        for name, loader in datasets.items():
            self.register(name, loader)

    def defer(self, callback: Callable):
        self._deffered.append(callback)
    
    @property
    def datasets(self):
        while self._deffered:
            cb = self._deffered.pop(0)
            self.register_all(cb(self))
        return self._datasets

    def load(self, name: str, **kwargs):
        if not name in self.datasets:
            raise ValueError(f"Unknown dataset {name}")
        return self.datasets[name](**kwargs)

def deferred_module_registry(module_name, registry_name='registry'):
    def cb(registry):
        logger.debug(f"Loading datasets from {module_name}")
        import inspect
        import importlib
        frm = inspect.stack()[1]
        pkg = inspect.getmodule(frm[0]).__name__
        mod = importlib.import_module(module_name, package=pkg)
        if not hasattr(mod, registry_name):
            return
        registry = getattr(mod, registry_name)
        return registry
    return cb

image_label_datasets = Registry() # type: Registry[Dataset]
image_label_datasets.defer(deferred_module_registry(".mnist"))