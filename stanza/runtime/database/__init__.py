import numpy as np
import jax.numpy as jnp
import urllib
import random

from stanza.util.dataclasses import dataclass
from stanza.util.logging import logger
from typing import Any
from pathlib import Path

NOUNS_ = None
ADJECTIVES_ = None
def words_():
    global NOUNS_
    global ADJECTIVES_
    if NOUNS_ is None:
        nouns_path = Path(__file__).parent / "nouns.txt"
        NOUNS_ = open(nouns_path).read().splitlines()
    if ADJECTIVES_ is None:
        adjectives_path = Path(__file__).parent / "adjectives.txt"
        ADJECTIVES_ = open(adjectives_path).read().splitlines()
    return ADJECTIVES_, NOUNS_

@dataclass(frozen=True)
class Figure:
    fig: Any

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

    def create(self):
        length = len(self.children)
        while True:
            adjectives, nouns = words_()
            adjective = random.choice(adjectives)
            noun = random.choice(nouns)
            name = f"{adjective}-{noun}-{length + 1}"
            if self.has(name):
                continue
            return self.open(name)

    def open(self, name):
        pass

    def add(self, name, value):
        pass

    # open a root-level table
    # name will be auto-generated if None
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