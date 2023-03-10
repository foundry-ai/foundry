import numpy as np
from stanza.util.logging import logger
from dataclasses import dataclass

import jax.numpy as jnp
import urllib

class Figure:
    def __init__(self, fig):
        self.fig = fig

class Video:
    def __init__(self, data, fps=4):
        self.data = data
        self.fps = fps

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
        elif parsed.scheme == 'wandb':
            entity = parsed.path.lstrip('/')
            from stanza.runtime.database.wandb import WandbDatabase
            return WandbDatabase(entity)
        else:
            raise RuntimeError("Unknown database url")

class Table:
    # open a sub-table
    # name will be auto-generated if None
    def open(self, name=None):
        pass

    def log(self, data):
        pass

    def tag(self, name):
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