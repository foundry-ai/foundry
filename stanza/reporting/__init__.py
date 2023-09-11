import numpy as np
import jax.numpy as jnp
import urllib
import random

from stanza.dataclasses import dataclass
from stanza.util.logging import logger
from typing import Any
from pathlib import Path


@dataclass(frozen=True)
class Figure:
    fig: Any
    height: int = None
    width: int = None

@dataclass(frozen=True)
class Video:
    data: np.array
    fps: int = 28

class Database:
    # The name of this database
    # in the parent (or in case of the
    # root, a descriptive name)
    @property
    def name(self):
        return None

    # Get the parent database
    # None for the root
    @property
    def parent(self):
        return None

    # should return a set of
    # keys
    @property
    def children(self):
        return set()

    def has(self, name):
        pass

    def __contains__(self, name):
        return self.has(name)

    def open(self, name):
        pass

    def create(self):
        return self.open(name=None)

    # must have stream=True
    # to log over steps
    def add(self, name, value, *,
            append=False, step=None, batch=False):
        raise NotImplementedError()

    # if batch=True, this is a whole 
    # batch of steps which we should log
    def log(self, data, *, step=None, batch=False):
        for (k,v) in flat_items(data):
            self.add(k,v, append=True, step=step, batch=batch)

    # open a root-level table
    # name will be auto-generated if None
    @staticmethod
    def from_url(db_url):
        parsed = urllib.parse.urlparse(db_url)
        if parsed.scheme == 'dummy':
            from stanza.reporting.dummy import DummyDatabase
            return DummyDatabase()
        elif parsed.scheme == 'local':
            from stanza.reporting.local import LocalDatabase
            return LocalDatabase()
        elif parsed.scheme == 'wandb':
            entity = parsed.path.lstrip('/')
            from stanza.reporting.wandb import WandbDatabase
            return WandbDatabase(entity)
        else:
            raise RuntimeError("Unknown database url")

def flat_items(d, prefix=''):
    for (k,v) in d.items():
        if isinstance(v, dict):
            yield from flat_items(v, prefix=f'{prefix}{k}.')
        else:
            yield (f'{prefix}{k}',v)

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