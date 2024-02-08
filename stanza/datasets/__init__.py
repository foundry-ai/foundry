from typing import Callable, Generic, TypeVar, Optional, Union, Protocol, Iterable, Tuple

from stanza import struct
from stanza.data import Data

import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@struct.dataclass(frozen=True)
class Dataset(Generic[T]):
    splits: dict[str, Data[T]] = struct.field(default_factory=dict)

DatasetBuilder = Callable[[], Dataset]

class DatasetBuilders(Protocol):
    def items(self) -> Iterable[Tuple[str, DatasetBuilder]]: ...

class DatasetRegistry(Generic[T]):
    def __init__(self):
        self._datasets = {}
        self._deffered = []
    
    def register(self, name: str, loader: Callable, *, transform: Optional[Callable] = None):
        if transform is not None:
            loader = transform(loader)
        self._datasets[name] = loader
    
    def register_all(self, datasets: Optional[DatasetBuilders], *, transform: Optional[Callable] = None):
        if datasets is None:
            return
        for name, loader in datasets.items():
            self.register(name, loader, transform=transform)

    def defer(self, callback: Callable):
        self._deffered.append(callback)
    
    @property
    def datasets(self) -> dict[str, Callable]:
        while self._deffered:
            cb = self._deffered.pop(0)
            self.register_all(cb(self))
        return self._datasets
    
    def items(self) -> Iterable[Tuple[str, DatasetBuilder]]:
        return self.datasets.items()

    def load(self, name: str, **kwargs) -> T:
        if not name in self.datasets:
            raise ValueError(f"Unknown dataset {name}")
        return self.datasets[name](**kwargs)

def register_module(module_name, registry_name='dataset_registry'):
    import inspect
    import importlib
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def cb(registry):
        mod = importlib.import_module(module_name, package=pkg)
        if not hasattr(mod, registry_name):
            return
        registry = getattr(mod, registry_name)
        return registry
    return cb

image_label_datasets = DatasetRegistry() # type: DatasetRegistry[Dataset]
image_label_datasets.defer(register_module(".mnist"))