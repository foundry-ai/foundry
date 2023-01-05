# The Aim ML backend for the experiment logger
from jinx.experiment import Lab, Experiment, Scalar, Figure

from aim import Repo, Run
import aim
import numpy as np
import jax.numpy as jnp
import dataclasses

from random_word import RandomWords

rw = RandomWords()

class AimScalar(Scalar):
    def __init__(self, run, name):
        self.run = run
        self.name = name
    
    def _log(self, value):
        self.run.track(value, self.name)

class AimFigure(Figure):
    def __init__(self, run, name):
        self.run = run
        self.name = name
    
    def _log(self, value):
        self.run.track(aim.Figure(value), self.name)

class AimLab(Lab):
    def __init__(self, repo=None):
        self.repo = Repo(repo)
    
    def create(self, name=None):
        first = rw.get_random_word()
        second = rw.get_random_word()
        return AimExperiment(repo=self.repo, exp=name,
                            name=f'{first}-{second}')

class AimExperiment(Experiment):
    def __init__(self, run=None,
                repo=None, hash=None, exp=None, name=None,
                root=True, prefix=''):
        self.run = Run(
            repo=repo,
            capture_terminal_logs=False,
            log_system_params=False,
            system_tracking_interval=None,
            experiment=exp
        ) if run is None else run
        if name is not None:
            self.run.name = name
        self.root = root
        self.prefix = prefix
    
    @property
    def name(self):
        return self.run.experiment
    
    @property
    def hash(self):
        return self.run.run_hash
    
    # set experiment parameter
    def __delitem__(self, val):
        del self.run[f'{self.prefix}{val}']

    def __getitem__(self, val):
        return self.run[f'{self.prefix}{val}']

    def __setitem__(self, name, val):
        self.run[f'{self.prefix}{name}'] = val
    
    def remove_tag(self, tag):
        self.run.remove_tag(tag)

    def add_tag(self, tag):
        self.run.add_tag(tag)

    def sub_experiment(self, name, **context):
        return AimExperiment(root=False,
                run=self.run,
                prefix=f'{name}/')

    def channel(self, name, type):
        if type is Scalar:
            return AimScalar(self.run, f'{self.prefix}{name}')
        elif type is Figure:
            return AimFigure(self.run, f'{self.prefix}{name}')

    def finish(self):
        self.run.report_successful_finish()