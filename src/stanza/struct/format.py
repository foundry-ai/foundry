import argparse

from typing import Any, Generic, TypeVar, Callable, Type, Protocol

T = TypeVar('T')
I = TypeVar('I')
O = TypeVar('O')

class FormatNotFoundError(ValueError): ...

class Context:
    def __init__(self, format : str, config: Any,
                    default_providers):
        self.format = format
        self.config = config
        self.default_providers = default_providers
    
    def with_config(self, config):
        return Context(self.format, config, self.default_providers)
    
    def format_for(self, t: Type, format: str = None) -> "Format":
        format = format or self.format
        if hasattr(t, f"__{format}_format__"):
            provider = getattr(t, f"__{format}_format__")
            return provider(self)
        if format in self.default_providers:
            return self.default_providers[format](self, t)
        raise FormatNotFoundError(f"No format {format} for {t}")

class FormatProvider(Protocol[T, I, O]):
    def __call__(self, ctx: Context, t: Type[T]) -> "Format[T, I, O]": ...

class Format(Generic[T, I, O]):
    def parser(self) -> Callable[[I], T]: ...
    def packer(self) -> Callable[[T], O]: ...