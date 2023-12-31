import jax
import optax
import jax.numpy as jnp

from stanza.train import Trainer
from typing import NamedTuple, Any, Tuple

from stanza.dataclasses import dataclass
from stanza.util.logging import logger

# Behavior cloning trainer
class BCTrainer:
    def __init__(self, net):
        pass

    # Will return a policy
    def run(self, dataset, rng_key):
        pass

# Diffusion Policy
class DPTrainer:
    def __init__(self, net):
        pass
    
    def run(self, dataset, rng_key):
        pass