import wandb

from jinx.experiment import Repo, Experiment, Run, remap, \
                            Video, Figure

import numpy as np

class WandbRepo:
    def __init__(self, entity):
        self.entity = entity
    
    def experiment(self, name):
        return WandbExperiment(self.entity, name)

class WandbExperiment(Experiment):
    def __init__(self, entity, name):
        self.entity = entity
        self.name = name
    
    def create_run(self, name=None):
        run = wandb.init(entity=self.entity, project=self.name, name=name)
        return WandbRun(run)

class WandbRun(Run):
    def __init__(self, run):
        self.run = run
    
    def _log(self, data):
        wandb.log(remap(data, {
            Figure: lambda f: f.fig,
            Video: lambda v: wandb.Video(np.array(v.data), fps=v.fps)
        }))