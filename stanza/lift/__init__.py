from typing import Any
import jax
import itertools

from stanza.lift.context import Context, Module

class HashProxyCallable:
    def __init__(self, value, *proxies):
        self.value = value
        self.proxies = proxies

    def __hash__(self):
        return hash(self.proxies)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.value(*args, **kwds)

def jit(f):
    return f