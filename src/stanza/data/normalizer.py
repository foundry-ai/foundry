from typing import Any, Generic, Callable, TypeVar
from stanza.struct import dataclass
from stanza.data import PyTreeData

import abc
import jax
import jax.numpy as jnp

T = TypeVar("T")
V = TypeVar("V")

class Normalizer(abc.ABC, Generic[T]):
    @property
    def structure(self) -> T: ...
    def map(self, fun : Callable[[T], V]) -> "Normalizer[V]": ...
    def normalize(self, data: T) -> T: ...
    def unnormalize(self, data: T) -> T: ...

@dataclass
class Chain(Generic[T], Normalizer[T]):
    normalizers : list[Normalizer[T]]

    @property
    def structure(self) -> T:
        return self.normalizers[0].structure

    def map(self, fun : Callable[[T], V]) -> "Chain[V]":
        return Chain(list([n.map(fun) for n in self.normalizers]))

    def normalize(self, data : T) -> T:
        for n in self.normalizers:
            data = n.normalize(data)
        return data

    def unnormalize(self, data : T) -> T:
        for n in reversed(self.normalizers):
            data = n.unnormalize(data)
        return data

@dataclass
class Compose(Generic[T], Normalizer[T]):
    normalizers: T # A T-shaped structer of normalizers

    @property
    def structure(self) -> T:
        return jax.tree_map(lambda x: x.structure,
            self.normalizers, is_leaf=lambda x: isinstance(x, Normalizer))
    
    def map(self, fun : Callable[[T], V]) -> "Compose[V]":
        # TODO: This doesn't work properly! Somehow construct an instance
        # of V from the normalizers, and deduce the mapping of the sub-normalizers
        # from the function output (?)
        # for now just do this...
        return Compose(fun(self.normalizers))
    
    def normalize(self, data : T) -> T:
        return jax.tree_map(lambda n, x: n.normalize(x),
            self.normalizers, data, is_leaf=lambda x: isinstance(x, Normalizer))
    
    def unnormalize(self, data : T) -> T:
        return jax.tree_map(lambda n, x: n.unnormalize(x),
            self.normalizers, data, is_leaf=lambda x: isinstance(x, Normalizer))

@dataclass
class ImageNormalizer(Normalizer[jax.Array]):
    """A simple normalizer which scales images from 0-255 (uint) to -1 -> 1 (float)"""
    _structure: jax.ShapeDtypeStruct

    @property
    def structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self._structure.shape, jnp.float32)

    def map(self, fun : Callable[[jax.Array], jax.Array]) -> "ImageNormalizer":
        return ImageNormalizer(fun(self.sample))

    def normalize(self, data : jax.Array) -> jax.Array:
        return data.astype(jnp.float32)/127.5 - 1.

    def unnormalize(self, data : jax.Array) -> jax.Array:
        return ((data + 1.)*127.5).astype(jnp.uint8)

# Will rescale to [-1, 1]
@dataclass
class LinearNormalizer(Generic[T], Normalizer[T]):
    min: T
    max: T

    @property
    def structure(self) -> T:
        return self.min

    def map(self, fun : Callable[[T], V]) -> "LinearNormalizer[V]":
        return LinearNormalizer(
            fun(self.min), fun(self.max)
        )

    def normalize(self, data : T) -> T:
        def norm(x, nmin, nmax):
            scaled = (x - nmin)/(nmax - nmin + 1e-6)
            # shift to [-1, 1]
            return 2*scaled - 1
        return jax.tree_util.tree_map(
            norm,
            data, self.min, self.max)

    def unnormalize(self, data : T) -> T:
        def unnorm(x, nmin, nmax):
            scaled = (x + 1)/2
            # shift to [nmin, nmax]
            return scaled*(nmax - nmin) + nmin
        return jax.tree_util.tree_map(
            unnorm,
            data, self.min, self.max)

    @staticmethod
    def from_data(data : T) -> "LinearNormalizer[T]":
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
class DummyNormalizer(Generic[T], Normalizer[T]):
    sample: T

    @property
    def structure(self) -> T:
        return self.sample
    
    def map(self, fun : Callable[[T], V]) -> "DummyNormalizer[V]":
        return DummyNormalizer(fun(self.sample))

    def normalize(self, data : T) -> T:
        return data

    def unnormalize(self, data : T) -> T:
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