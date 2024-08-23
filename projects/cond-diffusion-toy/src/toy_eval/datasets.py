import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Any
from stanza.dataclasses import dataclass, replace
from stanza.datasets import Dataset

@dataclass
class Sample:
    cond: Any
    value: Any

@dataclass
class TwoDeltasConfig:
    train_data_size: int = 64
    test_data_size: int = 32
    dim: int = 2
    weight: float = 1.0
    shift: float = 0.0

@dataclass
class TwoDeltaSequenceConfig:
    train_data_size: int = 64
    test_data_size: int = 32
    dim: int = 1
    sequence_length: int = 4
    weight: float = 1.0
    shift: float = 0.0

def create(config, rng_key: PRNGKey):
    if isinstance(config, TwoDeltasConfig):

        a = -jnp.ones((config.dim,)) / jnp.sqrt(config.dim)
        b = jnp.ones((config.dim,)) / jnp.sqrt(config.dim)
        deltas = jnp.stack([a, b])
        
        conds = jnp.array([0, config.shift, 1-config.shift, 1])
        p = jnp.array([[1,0],[config.weight, 1-config.weight],[1-config.weight, config.weight],[0,1]])
        
        def generate(rng_key, conds, p):
            i_rng, j_rng = jax.random.split(rng_key, 2)
            i = jax.random.choice(i_rng, conds.shape[0])
            j = jax.random.choice(j_rng, 2, p=p[i,:])
            return Sample(jnp.array([conds[i]]), jnp.array(deltas[j]))
        
        train = jax.vmap(generate, in_axes=[0, None, None])(
                jax.random.split(rng_key, config.train_data_size), conds, p)
        # test.values zero placeholder
        test = Sample(jnp.linspace(0, 1, config.test_data_size), jnp.zeros((config.test_data_size, config.dim)))

    elif isinstance(config, TwoDeltaSequenceConfig):

        a = -jnp.repeat(jnp.linspace(0, 1, config.sequence_length)[:,None], config.dim, axis=-1) / jnp.sqrt(config.dim)
        b = jnp.repeat(jnp.linspace(0, 1, config.sequence_length)[:,None], config.dim, axis=-1) / jnp.sqrt(config.dim)
        print(b)
        deltas = jnp.stack([a, b])
        
        conds = jnp.array([0, config.shift, 1-config.shift, 1])
        p = jnp.array([[1,0],[config.weight, 1-config.weight],[1-config.weight, config.weight],[0,1]])
        
        def generate(rng_key, conds, p):
            i_rng, j_rng = jax.random.split(rng_key, 2)
            i = jax.random.choice(i_rng, conds.shape[0])
            j = jax.random.choice(j_rng, 2, p=p[i,:])
            return Sample(jnp.array([conds[i]]), jnp.array(deltas[j]))
        
        train = jax.vmap(generate, in_axes=[0, None, None])(
                jax.random.split(rng_key, config.train_data_size), conds, p)
        # test.values zero placeholder
        test = Sample(jnp.linspace(0, 1, config.test_data_size), jnp.zeros((config.test_data_size, config.dim)))
    else:
        raise ValueError(f"Unknown dataset: {config}")
    
    return Dataset(
        splits={"train": train, "test": test},
        normalizers={},
        transforms={}
    )