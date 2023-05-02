from typing import Any
from stanza.util.dataclasses import dataclass

import jax
import jax.numpy as jnp

# Will rescale to [-1, 1]
@dataclass(jax=True)
class LinearNormalizer:
    min: Any
    max: Any

    def normalize(self, data):
        def norm(x, nmin, nmax):
            scaled = (x - nmin)/(nmax - nmin)
            # shift to [-1, 1]
            return 2*scaled - 1
        return jax.tree_util.tree_map(
            norm,
            data, self.min, self.max)

    def unnormalize(self, data):
        def unnorm(x, nmin, nmax):
            scaled = (x + 1)/2
            # shift to [nmin, nmax]
            return scaled*(nmax - nmin) + nmin
        return jax.tree_util.tree_map(
            unnorm,
            data, self.min, self.max)

    @staticmethod
    def from_data(data):
        min = jax.util.tree_map(
            lambda x: jnp.min(x, axis=0), data.data
        )
        max = jax.util.tree_map(
            lambda x: jnp.max(x, axis=0), data.data
        )
        return LinearNormalizer(min, max)