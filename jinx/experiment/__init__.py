import numpy as np
from jinx.logging import logger

import jax.numpy as jnp

# TODO: No idea what this API should look like yet

# Type hint annotations
class Figure:
    def __init__(self, fig):
        self.fig = fig

class Video:
    def __init__(self, data, fps=4):
        self.data = data
        self.fps = fps

class Repo:
    def experiment(self, name):
        pass

    @staticmethod
    def from_url(repo_url):
        if repo_url == 'dummy':
            from jinx.experiment.dummy import DummyRepo
            return DummyRepo()
        elif repo_url.startswith('wandb/'):
            entity = repo_url[6:]
            from jinx.experiment.wandb import WandbRepo
            return WandbRepo(entity)

class Experiment:
    def create_run(self, name=None):
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

# Helper function to merge 
def _merge(a, b, path=None):
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

class Run:
    def __init__(self):
        self.step = 0
        self._temp_data = {}

    def log(self, data, step=None, commit=True):
        if step is None and commit:
            self._log(data)
        elif step == self.step or not commit:
            merge(self._temp_data, data)
        elif step == self.step + 1:
            self._log(self._temp_data)
            self._temp_data = {}
            merge(data, self._temp_data)
        else:
            raise RuntimeError("Step must be either current or next!")

    def _log(self, data):
        raise RuntimeError("Not implemented!")
