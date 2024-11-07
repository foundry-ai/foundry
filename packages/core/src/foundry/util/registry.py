import typing
from typing import Any, Generic, Iterable

T = typing.TypeVar('T')

class Builder(typing.Protocol[T]):
    def __call__(self, **kwargs: dict[str, Any]) -> T: ...

class Registry(Generic[T]):
    def __init__(self):
        self._registry : dict[str, Builder[T]] = {}
    
    def extend(self, registry, prefix=None):
        if hasattr(registry, "entries"):
            items = registry.entries()
        else:
            items = registry.items()
        for path, builder in items:
            if prefix is not None:
                path = f"{prefix}/{path}"
            self.register(path, builder)

    def register(self, path: str, builder: Builder[T], prefix=None):
        if prefix:
            path = f"{prefix}{path}"
        if path in self._registry:
            raise ValueError(f"Path {path} already registered")
        self._registry[path] = builder

    def create(self, path: str, /, **kwargs) -> T:
        if path not in self._registry:
            raise ValueError(f"{path} not found!")
        return self._registry[path](**kwargs)
    
    def keys(self) -> Iterable[str]:
        return list(self._registry.keys())

    def entries(self) -> Iterable[tuple[str, Builder[T]]]:
        return self._registry.items()

    def __call__(self, path: str, /, **kwargs) -> T:
        return self.create(path, **kwargs)
    