from stanza.data import Data
from stanza.dataclasses import dataclass, field
from typing import Callable, Any
from jax.random import PRNGKey

import jax

@dataclass(jax=True)
class Validator:
    rng_key: PRNGKey
    dataset: Data
    batch_size: int = field(default=16, jax_static=True)
    # set to None to run through the entire validation dataset
    samples_per_run: int = field(default=None)

    def init(self, state):
        return self.rng_key

    def __call__(self, rng_key, state):
        rng_key, sk = jax.random.split(rng_key)
        dataset = self.dataset.shuffle(sk)
        dataset = dataset.batch(self.batch_size)

        def scan_fn(carry, batch):
            running_stats, total = carry
            _, _, stats = state.loss_fn(
                state.fn_params, 
                state.fn_state,
                rng_key, batch
            )
            new_total = total + self.batch_size
            running_stats = jax.tree_util.tree_map(
                lambda x, y: x*(total/new_total) + y*(self.batch_size/new_total),
                running_stats, stats
            )
            return (running_stats, new_total)
        stats, _ = dataset.scan(scan_fn, (state.last_stats, 0), limit=self.samples_per_run//self.batch_size)
        return rng_key, stats