from stanza.data import Data, PyTreeData
from stanza.dataclasses import dataclass, field, replace
from jax.random import PRNGKey
from typing import Callable
from stanza.util.loop import every_epoch

import jax.numpy as jnp
import jax
import chex

@dataclass(jax=True)
class Validator:
    rng_key: PRNGKey
    dataset: Data
    condition : Callable = every_epoch
    stat_fn : Callable = None
    stat_key : str = field(default="test", jax_static=True)
    batch_size: int = field(default=None, jax_static=True)
    # set to None to run through the entire validation dataset
    samples_per_run: int = field(default=None)

    def init(self, state):
        sample = self.dataset[0]
        batch = jax.tree_map(lambda x: jnp.expand_dims(x, 0), sample)

        if self.stat_fn is None:
            _, _, stats = state.config.loss_fn(
                state.fn_state,
                state.fn_params, 
                state.rng_key, batch
            )
        else:
            stats = self.stat_fn(state.fn_state, 
                    state.fn_params, state.rng_key, batch)
        last_stats = dict(state.last_stats)
        last_stats[self.stat_key] = stats
        state = replace(state, last_stats=last_stats)
        return (self.rng_key, state.iteration), state

    def __call__(self, hs, state):
        _, iteration = hs
        stat_fn = self.stat_fn if self.stat_fn is not None else \
            lambda s, p, r, b: state.config.loss_fn(s,p,r,b)[2]
        def cond_fn(hs, state):
            rng_key, _ = hs
            rng_key, sk = jax.random.split(rng_key)

            if self.batch_size is not None:
                dataset = self.dataset.shuffle(sk)
                dataset = dataset.batch(self.batch_size)
                def scan_fn(carry, batch):
                    batch = PyTreeData.from_data(batch).data
                    running_stats, total = carry
                    stats = stat_fn(
                        state.fn_state,
                        state.fn_params, 
                        rng_key, batch
                    )
                    new_total = total + self.batch_size
                    running_stats = jax.tree_util.tree_map(
                        lambda x, y: x*(total/new_total) + y*(self.batch_size/new_total),
                        running_stats, stats
                    )
                    return (running_stats, new_total)
                batches = (self.samples_per_run // self.batch_size) \
                    if self.samples_per_run is not None else None
                stats, _ = dataset.scan(scan_fn, (state.last_stats["test"], 0), limit=batches)
            else:
                data = PyTreeData.from_data(self.dataset).data
                stats = stat_fn(
                    state.fn_state,
                    state.fn_params, 
                    rng_key, data
                )
            chex.assert_trees_all_equal_shapes_and_dtypes(
                state.last_stats[self.stat_key], stats)
            
            last_stats = dict(state.last_stats)
            last_stats[self.stat_key] = stats
            state = replace(state, last_stats=last_stats)
            return (rng_key, state.iteration), state
        return jax.lax.cond(jnp.logical_and(self.condition(state),
                            iteration != state.iteration),
            cond_fn, lambda x, y: (x,y), hs, state)