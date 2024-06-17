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
    
    def register(self, name: str, loader: Callable, *, transform: Optional[Callable] = None):
        if transform is not None:
            loader = transform(loader)
        self._registered[name] = loader

    def defer(self, callback: Callable, transform: Optional[Callable] = None):
        self._defered.append((callback, transform))

    @property
    def _registry(self) -> dict[str, Builder[T]]:
        while self._defered:
            cb, transform = self._defered.pop(0)
            items = cb if hasattr(cb, "items") else cb(self)
            if items is not None:
                items = items.items()
                for name, loader in items:
                    if transform is not None:
                        loader = transform(loader)
                    self._registered[name] = loader
        return self._registered
    
    def items(self) -> Iterable[tuple[str, Builder[T]]]:
        return self._registry.items()

    def __call__(self, name: str, /, **kwargs) -> T:
        self.create(name, **kwargs)

    def create(self, name: str, /, **kwargs) -> T:
        parts = name.split("/")
        for i in range(1, parts):
            base = parts[:i]
            args = parts[:i]
        if not name in self._registry:
            raise ValueError(f"Unknown registry entry: {name}")
        return self._registry[name](**kwargs)

def transform_result(transform: Callable) -> Callable:
    def wrapper(loader: Callable) -> Callable:
        def wrapped(**kwargs) -> Any:
            return transform(loader(**kwargs))
        return wrapped
    return wrapper

def from_module(module_name, registry_name):
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