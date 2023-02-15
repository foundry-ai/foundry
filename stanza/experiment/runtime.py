import os
import pickle
import argparse
import functools

from .config import add_to_parser, from_args
from dataclasses import dataclass
from stanza.logging import logger

@dataclass
class EmptyConfig:
    pass

# An acivity reports its data to an experiment
class Activity:
    def __init__(self, config_dataclass=None):
        self.config_dataclass = config_dataclass
    
    def run(self, config=None, experiment=None):
        pass

class FuncActivity(Activity):
    def __init__(self, config_dataclass, exec):
        super().__init__(name, config_dataclass)
        self._exec = exec
    
    def run(self, config=None, *args, **kwargs):
        if config is None:
            parser = argparse.ArgumentParser()
            add_to_parser(self.config_dataclass, parser)
            cargs = parser.parse_args()
            config = config.from_args(self.config_dataclass, cargs)
        return self._exec(config, *args, **kwargs)

    def __call__(self, config=None, *args, **kwargs):
        return self.run(config, *args, **kwargs)

# A decorator version for convenience
def activity(config_dataclass=None, f=None):
    if callable(f):
        return FuncActivity(config_dataclass, f)

    # return the decorator
    def decorator(f):
        a = FuncActivity(config_dataclass, f)
        return functools.wraps(f)(a)
    return decorator