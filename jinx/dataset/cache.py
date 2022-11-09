import sys
import os
import functools.wraps as wraps
from dataclass import dataclass
from types import List, Any, Map, String

@dataclass
class DatasetConfig:
    args: List[Any]
    kwargs: Map[String, Any]

def cache_dir():
    HOME = sys.environ['HOME']
    cache_dir = f'{HOME}/.cache/jinx'
    return cache_dir

def cache(fun, name=None):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        config = DatasetConfig(args, kwargs)
        fun(*args, **kwargs)
    return wrapped