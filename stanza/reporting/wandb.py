import wandb
import jax

from stanza.reporting import Database, \
                            Video, Figure, remap

import numpy as np

class WandbDatabase(Database):
    def __init__(self, path):
        path = path or "dpfrommer-projects/"
        entity, project = path.split("/")
        self.entity = entity
        self.project = project
    
    def open(self, name=None):
        run = wandb.init(entity=self.entity, project=self.project, name=name)
        return WandbRun(run)
    
    def add(self, name, value, append=False):
        # TODO: Make automatically open some default name run
        # if adding to the top level database
        raise RuntimeError("Cannot add to root-level wandb database!")

class WandbRun(Database):
    def __init__(self, run, prefix=''):
        self.run = run
        self.prefix = prefix

    def open(self, name):
        n = f"{self.prefix}.{name}"
        return WandbRun(self.run, n)

    def add(self, name, value, append=False):
        pass

    def flush(self):
        pass

    def log(self, data, batch=False):
        from stanza.util import shape_tree
        data = remap(data, {
                Figure: lambda f: f.fig,
                Video: lambda v: wandb.Video(np.array(v.data), fps=v.fps)
            })
        if batch:
            dim = jax.tree_util.tree_leaves(data)[0].shape[0]
            for i in range(dim):
                x = jax.tree_map(lambda x: x[i], data)
                if self.prefix != '':
                    self.run.log({self.prefix: x})
                else:
                    self.run.log(x)
                self.run.log(x)
        else:
            if self.prefix != '':
                self.run.log({self.prefix: data})
            else:
                self.run.log(data)