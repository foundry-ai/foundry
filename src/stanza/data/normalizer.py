from typing import Any
from stanza.struct import dataclass, field
from stanza.data import PyTreeData

import jax
import jax.numpy as jnp

Normalizer = Any

# Will rescale to [-1, 1]
@dataclass
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
        min = jax.tree_util.tree_map(
            lambda x: jnp.min(x, axis=0), data.tree
        )
        max = jax.tree_util.tree_map(
            lambda x: jnp.max(x, axis=0), data.tree
        )
        return LinearNormalizer(min, max)

@dataclass
class DummyNormalizer:
    sample: Any

    @property
    def instance(self):
        return self.sample

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data

@dataclass
class StdNormalizer:
    mean: Any = None
    var: Any = None
    count: int = 0
    std: Any = None

    @property
    def instance(self):
        return self.mean

    def map(self, fun):
        return StdNormalizer(
            fun(self.mean), fun(self.var),
            self.count, fun(self.std)
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
        batch_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), batch)
        batch_var = jax.tree_map(lambda x: jnp.var(x, axis=0), batch)

        if self.var is None:
            return StdNormalizer(batch_mean, batch_var, n)
        total = self.count + n
        mean_delta = jax.tree_map(lambda x, y: x - y,
                                  batch_mean, self.mean)
        new_mean = jax.tree_map(lambda x, y: x + n/total * y,
                                self.mean, mean_delta)

        m_a = jax.tree_map(lambda v: v*self.total, self.var)
        m_b = jax.tree_map(lambda v: v*n, batch_var)
        m2 = jax.tree_map(
            lambda a, b, d: a + b + d * n * self.count / total,
            m_a, m_b, mean_delta
        )
        new_var = jax.tree_map(lambda x: x/total, m2)
        return StdNormalizer(new_mean, new_var, total)

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
    
    @staticmethod
    def empty_for(sample):
        zeros = jax.tree_map(lambda x: jnp.zeros_like(x), sample)
        ones = jax.tree_map(lambda x: jnp.ones_like(x), sample)
        return StdNormalizer(zeros, ones, jnp.zeros(()))