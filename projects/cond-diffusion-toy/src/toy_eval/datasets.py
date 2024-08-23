import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Any
from stanza.dataclasses import dataclass, replace
from stanza.runtime import ConfigProvider
from stanza.datasets import Dataset

@dataclass
class Sample:
    cond: Any
    value: Any

@dataclass
class TwoDeltasConfig:
    train_data_size: int = 64
    test_data_size: int = 64
    dim: int = 2

    def parse(self, config: ConfigProvider) -> "TwoDeltasConfig":
        return config.get_dataclass(self)

@dataclass
class TwoDeltaSequenceConfig:
    train_data_size: int = 64
    test_data_size: int = 64
    dim: int = 1
    sequence_length: int = 4

    def parse(self, config: ConfigProvider) -> "TwoDeltaSequenceConfig":
        return config.get_dataclass(self)

def create(config, rng_key: PRNGKey):
    if isinstance(config, TwoDeltasConfig):

        a = -jnp.ones((config.dim,)) / jnp.sqrt(config.dim)
        b = jnp.ones((config.dim,)) / jnp.sqrt(config.dim)
        deltas = jnp.stack([a, b])
        
        conds = jnp.array([-1.0,1.0], dtype=jnp.float32)
        p = jnp.array([[1.0,0.0],[0.0,1.0]], dtype=jnp.float32)
        
        def generate(rng_key, conds, p):
            i_rng, j_rng = jax.random.split(rng_key, 2)
            i = jax.random.choice(i_rng, conds.shape[0])
            j = jax.random.choice(j_rng, 2, p=p[i,:])
            return Sample(jnp.array([conds[i]]), jnp.array(deltas[j]))
        
        train = jax.vmap(generate, in_axes=[0, None, None])(
                jax.random.split(rng_key, config.train_data_size), conds, p)
        # test.values zero placeholder
        test = Sample(jnp.linspace(-1, 1, config.test_data_size), jnp.zeros((config.test_data_size, config.dim)))
        return Dataset(
            splits={"train": train, "test": test},
            normalizers={},
            transforms={"visualize": lambda x: jnp.dot(x, jnp.ones((config.dim,))/jnp.sqrt(config.dim))[...,None]}
        )

    elif isinstance(config, TwoDeltaSequenceConfig):

        a = -jnp.repeat(jnp.linspace(0.5, 1, config.sequence_length)[:,None], config.dim, axis=-1) / jnp.sqrt(config.dim)
        b = jnp.repeat(jnp.linspace(0.5, 1, config.sequence_length)[:,None], config.dim, axis=-1) / jnp.sqrt(config.dim)
        deltas = jnp.stack([a, b])
        
        conds = jnp.array([-1.0,1.0], dtype=jnp.float32)
        p = jnp.array([[1.0,0.0],[0.0,1.0]], dtype=jnp.float32)
        
        def generate(rng_key, conds, p):
            i_rng, j_rng = jax.random.split(rng_key, 2)
            i = jax.random.choice(i_rng, conds.shape[0])
            j = jax.random.choice(j_rng, 2, p=p[i,:])
            return Sample(jnp.array([conds[i]]), jnp.array(deltas[j]))
        
        train = jax.vmap(generate, in_axes=[0, None, None])(
                jax.random.split(rng_key, config.train_data_size), conds, p)
        # test.values zero placeholder
        test = Sample(jnp.linspace(-1, 1, config.test_data_size), jnp.zeros((config.test_data_size, config.dim)))
        return Dataset(
            splits={"train": train, "test": test},
            normalizers={},
            transforms={}
        )
    else:
        raise ValueError(f"Unknown dataset: {config}")
    
    