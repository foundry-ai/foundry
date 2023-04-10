
from typing import List, Any

from .main import iterative_learning, Config as IterConfig
from .ilqr import ilqr_learning, Config as ILQRConfig
import dataclasses
from dataclasses import dataclass, field, replace

import pickle
import os

from jinx.experiment.runtime import activity, EmptyConfig
from jinx.experiment.dummy import DummyRun

from multiprocessing import Pool
from jax.random import PRNGKey
from jinx.logging import logger
import jax

import subprocess

x0_rngs = [42, 100, 123, 573, 69, 87838, 909, 29893121,
        73873832, 14897235]

def run_iterative(config):
    return iterative_learning(config, DummyRun())

def run_ilqr(config):
    return ilqr_learning(config, DummyRun())


@activity("iter_sweep", IterConfig)
def iter_sweep(config, exp):
    pass

@activity("ilqr_sweep", ILQRConfig)
def ilqr_sweep(config, exp):
    pass