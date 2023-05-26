import wandb

from stanza.runtime.database import Database, \
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

class WandbRun(Table):
    def __init__(self, run, prefix=''):
        self.run = run
        self.prefix = prefix
   
    def log(self, data):
        data = remap(data, {
                Figure: lambda f: f.fig,
                Video: lambda v: wandb.Video(np.array(v.data), fps=v.fps)
            })
        if self.prefix != '':
            self.run.log({self.prefix: data})
        else:
            self.run.log(data)