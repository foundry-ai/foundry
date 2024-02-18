import typing
from typing import Iterable, Callable, Any, Generic, Optional

T = typing.TypeVar('T')

class Builder(typing.Protocol[T]):
    def __call__(self, **kwargs: dict[str, Any]) -> T: ...

class Builders(typing.Protocol[T]):
    def items(self) -> Iterable[tuple[str, Builder]]: ...

class Registry(Generic[T]):
    def __init__(self):
        self._registered = {} # type: dict[str, Builder[T]]
        self._deffered = []
    
    def register(self, name: str, loader: Callable, *, transform: Optional[Callable] = None):
        if transform is not None:
            loader = transform(loader)
        self._registered[name] = loader
    
    def register_all(self, datasets: Optional[Builders], *, transform: Optional[Callable] = None):
        if datasets is None:
            return
        for name, loader in datasets.items():
            self.register(name, loader, transform=transform)

    def defer(self, callback: Callable):
        self._deffered.append(callback)
    
    @property
    def _registry(self) -> dict[str, Builder[T]]:
        while self._deffered:
            cb = self._deffered.pop(0)
            self.register_all(cb(self))
        return self._registered
    
    def items(self) -> Iterable[tuple[str, Builder[T]]]:
        return self._registry.items()

    def create(self, name: str, **kwargs) -> T:
        if not name in self._registry:
            raise ValueError(f"Unknown registry entry: {name}")
        return self._registry[name](**kwargs)

def register_module(module_name, registry_name):
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