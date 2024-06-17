import typing
from typing import Iterable, Callable, Any, Generic, Optional

T = typing.TypeVar('T')

class Builder(typing.Protocol[T]):
    def __call__(self, **kwargs: dict[str, Any]) -> T: ...

class BuilderSet(typing.Protocol[T]):
    def __call__(self, path: str, **kwargs: dict[str, Any]) -> T: ...
    def keys() -> Iterator[str]: ...

class SingleBuilder(BuilderSet[T], Generic[T]):
    def __init__(self, builder : Builder[T]):
        self.builder = builder

    def __call__(self, path: str, **kwargs: dict[str, Any]) -> T:
        assert path == ""
        return self.func(**kwargs)

    def keys() -> Iterator[str]:
        return iter([""])

class Registry(Generic[T], BuilderSet[T]):
    def __init__(self):
        self._registry : dict[str, Builder[T]] = {}

    def extend(self, path: str, builders: Builders):
        parts = path.split("/")
        if parts[-1] != "":
            parts.append("")
        registry = self._registry
        for p in parts[:-1]:
            if len(p) == 0:
                raise ValueError("Part cannot have zero length!")
            registry = registry.setdefault(p, {})
            if not isinstance(registry, dict):
                raise ValueError("Cannot override registry")
        registry[""] = builders

    def register(self, path: str, builder: Builder):
        self.extend(path, SingleBuilder(builder))

    def create(self, path: str, /, **kwargs) -> T:
        parts = path.split("/")
        if parts[-1] != "":
            parts.append("")
        if len(parts) == 0:
            raise ValueError("Must have non-zero length path")
        registry = self._registry
        for i, p in parts[:-1]:
            if p in registry:
                registry = registry[p]
            else:
                remainder = "/".join(parts[i+1:])
                break
        return registry[""](remainder, **kwargs)

    def __call__(self, path: str, /, **kwargs) -> T:
        return self.create(path, **kwargs)
    
    @staticmethod
    def _keys(map) -> Iterator[str]:
        for k, v in map.items():
            if isinstance(v, dict): keys = Registry._keys(v)
            else: keys = v.keys()
            for key in keys:
                if key:
                    yield f"{k}/{key}"
                else:
                    yield k

    def keys(self) -> Iterator[str]:
        return Registry._keys(self._registry)

def from_module(module_name, variable_name):
    import inspect
    import importlib
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def cb(*args, **kwargs):
        mod = importlib.import_module(module_name, package=pkg)
        if not hasattr(mod, registry_name):
            return
        value = getattr(mod, registry_name)
        return value(*args, **kwargs)
    return cb