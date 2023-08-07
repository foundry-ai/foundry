from typing import Any
from stanza.dataclasses import dataclass, field
from stanza.data import PyTreeData

import jax
import jax.numpy as jnp

# Will rescale to [-1, 1]
@dataclass(jax=True)
class LinearNormalizer:
    min: Any
    max: Any

    @property
    def instance(self):
        return self.min

    def map(self, fun):
        return LinearNormalizer(
            fun(self.min), fun(self.max)
        )

    def normalize(self, data):
        def norm(x, nmin, nmax):
            scaled = (x - nmin)/(nmax - nmin + 1e-6)
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
        # For simplicity must be a PyTreeData
        # Convert to PyTreeFormat
        data = PyTreeData.from_data(data)
        min = jax.tree_util.tree_map(
            lambda x: jnp.min(x, axis=0), data.data
        )
        max = jax.tree_util.tree_map(
            lambda x: jnp.max(x, axis=0), data.data
        )
        return LinearNormalizer(min, max)


@dataclass(jax=True, kw_only=True)
class StdNormalizer:
    mean: Any = None
    var: Any = None
    total: int = 0
    std: Any = field(init=False)

    @property
    def instance(self):
        return self.mean

    def __post_init__(self):
        std = jax.tree_map(lambda x: jnp.sqrt(x + 1e-8), self.var)
        object.__setattr__(self, 'std', std)

    def map(self, fun):
        return StdNormalizer(
            fun(self.mean), fun(self.var),
            self.total, fun(self.std)
        )

    def normalize(self, data):
        if self.mean is not None:
            return jax.tree_map(
                lambda d, m, s: (d - m) / s,
                data, self.mean, self.std
            )
        else:
            return jax.tree_map(
                lambda d, s: d / s,
                data, self.std
            )

    def unnormalize(self, data):
        if self.mean is not None:
            return jax.tree_map(
                lambda d, m, s: d*s + m,
                data, self.mean, self.std
            )
        else:
            return jax.tree_map(
                lambda d, s: d * s,
                data, self.std
            )
    
    def update(self, batch):
        # get the batch dimension size
        n = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
    
    @staticmethod
    def from_data(data):
        data = PyTreeData.from_data(data)
        mean = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), data.data
        )
        var = jax.tree_util.tree_map(
            lambda x: jnp.var(x, axis=0), data.data
        )
        return StdNormalizer(mean, var, data.length)