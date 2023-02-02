import os
import pickle
from dataclasses import dataclass
from stanza.logging import logger



@dataclass
class EmptyConfig:
    pass

# An acivity reports its data to an experiment
class Activity:
    def __init__(self, name, config_dataclass=None):
        self.name = name
        self.config_dataclass = config_dataclass
    
    def run(self, config, experiment):
        pass

class FuncActivity(Activity):
    def __init__(self, name, config_dataclass, exec):
        super().__init__(name, config_dataclass)
        self._exec = exec
    
    def run(self, config, *args, **kwargs):
        return self._exec(config, *args, **kwargs)

    def __call__(self, config, *args, **kwargs):
        return self._exec(config, *args, **kwargs)

# A decorator version for convenience
def activity(name, config_dataclass=None):
    def build_wrapper(f):
        return FuncActivity(name, config_dataclass, f)
    return build_wrapper