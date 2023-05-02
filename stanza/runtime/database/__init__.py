import numpy as np
import jax.numpy as jnp
import urllib

from stanza.util.dataclasses import dataclass
from stanza.util.logging import logger
from typing import Any


@dataclass(frozen=True)
class Figure:
    fig: Any

@dataclass(frozen=True)
class Video:
    data: np.array
    fps: int = 28

@dataclass(frozen=True)
class PyTree:
    data: Any

class Database:
    # open a root-level table
    # name will be auto-generated if None
    def open(self, name=None):
        pass

    @staticmethod
    def from_url(db_url):
        parsed = urllib.parse.urlparse(db_url)
        if parsed.scheme == 'dummy':
            from stanza.runtime.database.dummy import DummyDatabase
            return DummyDatabase()
        elif parsed.scheme == 'local':
            from stanza.runtime.database.local import LocalDatabase
            return LocalDatabase()
        elif parsed.scheme == 'wandb':
            entity = parsed.path.lstrip('/')
            from stanza.runtime.database.wandb import WandbDatabase
            return WandbDatabase(entity)
        else:
            raise RuntimeError("Unknown database url")

class TableInfo:
    def __init__(self, table):
        self._table = table
    
    def __setattr__(self, name: str, value: Any):
        if name == '_table':
            super().__setattr__(name, value)
        else:
            self._table.set(name, value)
    
    def __getattr__(self, name: str) -> Any:
        return self._table.get(name)

class Table:
    # open a sub-table
    # name will be auto-generated if None
    def open(self, name=None):
        pass

def remap(obj, type_mapping):
    if isinstance(obj, dict):
        return { k: remap(v, type_mapping) for (k,v) in obj.items() }
    elif isinstance(obj, list):
        return [ remap(v, type_mapping) for v in obj ]
    elif isinstance(obj, tuple):
        return tuple([ remap(v, type_mapping) for v in obj ])
    elif isinstance(obj, jnp.ndarray):
        return np.array(obj)
    elif type(obj) in type_mapping:
        return type_mapping[type(obj)](obj)
    else:
        return obj