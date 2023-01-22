import wandb

from jinx.experiment import Repo, Experiment, Run, remap, \
                            Video, Figure

import numpy as np

class WandbRepo(Repo):
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
    def __init__(self, run, prefix=''):
        self.run = run
        self.prefix = prefix
   
    @property
    def config(self):
        return self.run.config
   
    @property
    def summary(self):
        return self.run.summary 
    
    def sub_run(self, prefix):
        if self.prefix != '':
            return WandbRun(self.run, f'{self.prefix}.{prefix}')
        else:
            return WandbRun(self.run, f'{prefix}')
    
    def _log(self, data):
        data = remap(data, {
                Figure: lambda f: f.fig,
                Video: lambda v: wandb.Video(np.array(v.data), fps=v.fps)
            })
        if self.prefix != '':
            self.run.log({self.prefix: data})
        else:
            self.run.log(data)