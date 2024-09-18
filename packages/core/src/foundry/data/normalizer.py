from typing import Any, Generic, Callable, TypeVar

import jax.flatten_util

from foundry.core import tree
from foundry.core.dataclasses import dataclass, field
from foundry.data import PyTreeData

import abc
import jax
import foundry.numpy as jnp

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
        return self.normalizers[-1].structure

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
        return tree.map(lambda x: x.structure,
            self.normalizers, is_leaf=lambda x: isinstance(x, Normalizer))
    
    def map(self, fun : Callable[[T], V]) -> "Compose[V]":
        mapped = fun(self.normalizers)
        # if we are left with a single normalizer, just return it
        if isinstance(mapped, Normalizer):
            return mapped
        return Compose(fun(self.normalizers))
    
    def normalize(self, data : T) -> T:
        return tree.map(lambda n, x: n.normalize(x),
            self.normalizers, data, is_leaf=lambda x: isinstance(x, Normalizer))
    
    def unnormalize(self, data : T) -> T:
        return tree.map(lambda n, x: n.unnormalize(x),
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
        return jax.tree.map(
            norm,
            data, self.min, self.max)

    def unnormalize(self, data : T) -> T:
        def unnorm(x, nmin, nmax):
            scaled = (x + 1)/2
            # shift to [nmin, nmax]
            return scaled*(nmax - nmin) + nmin
        return jax.tree.map(
            unnorm,
            data, self.min, self.max)

    @staticmethod
    def from_data(data : T) -> "LinearNormalizer[T]":
        # For simplicity must be a PyTreeData
        # Convert to PyTreeFormat
        min = jax.tree.map(
            lambda x: jnp.min(x, axis=0), data.tree
        )
        max = jax.tree.map(
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

@dataclass(kw_only=True)
class StdNormalizer:
    mean: Any = None
    var: Any = None
    std: Any = None
    count: int = 0

    @property
    def structure(self):
        return self.mean

    def map(self, fun):
        return StdNormalizer(
            fun(self.mean), fun(self.var),
            fun(self.std), self.count,
        )

    def normalize(self, data):
        if self.mean is not None:
            return tree.map(
                lambda d, m, s: (d - m) / (s + 1e-6),
                data, self.mean, self.std 
            )
        else:
            return tree.map(
                lambda d, s: d / (s + 1e-6),
                data, self.std 
            )

    def unnormalize(self, data):
        if self.mean is not None:
            return tree.map(
                lambda d, m, s: d * (s + 1e-6) + m,
                data, self.mean, self.std 
            )
        else:
            return tree.map(
                lambda d, s: d * (s + 1e-6),
                data, self.std 
            )
    
    def update(self, batch):
        # get the batch dimension size
        n = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
        batch_mean = tree.map(lambda x: jnp.mean(x, axis=0), batch)
        batch_var = tree.map(lambda x: jnp.var(x, axis=0), batch)

        if self.var is None:
            return StdNormalizer(batch_mean, batch_var, n)
        total = self.count + n
        mean_delta = tree.map(lambda x, y: x - y,
                                  batch_mean, self.mean)
        new_mean = tree.map(lambda x, y: x + n/total * y,
                                self.mean, mean_delta)

        m_a = tree.map(lambda v: v*self.total, self.var)
        m_b = tree.map(lambda v: v*n, batch_var)
        m2 = tree.map(
            lambda a, b, d: a + b + d * n * self.count / total,
            m_a, m_b, mean_delta
        )
        new_var = tree.map(lambda x: x/total, m2)
        new_std = tree.map(lambda x: jnp.sqrt(new_var), new_var)
        return StdNormalizer(new_mean, new_var, new_std, total)

    @staticmethod
    def from_data(data, component_wise=True):
        data = data.as_pytree()
        data_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(data)
        unflatten = jax.flatten_util.ravel_pytree(tree.map(lambda x: x[0], data))[1]
        if component_wise:
            mean = jnp.mean(data_flat, axis=0)
            var = jnp.var(data_flat, axis=0)
            std = jnp.sqrt(var)
            return StdNormalizer(
                unflatten(mean), unflatten(var),
                unflatten(std), data_flat.shape[0]
            )
        else:
            mean = jnp.mean(data_flat, axis=0)
            var = jnp.var(jnp.linalg.norm(data_flat - mean[None,:], axis=-1), axis=0)
            var = var*jnp.ones_like(data_flat[0])
            std = jnp.sqrt(var)
            return StdNormalizer(
                unflatten(mean), unflatten(var), 
                unflatten(std), data_flat.shape[0]
            )

    @staticmethod
    def empty_for(sample):
        zeros = tree.map(lambda x: jnp.zeros_like(x), sample)
        ones = tree.map(lambda x: jnp.ones_like(x), sample)
        return StdNormalizer(zeros, ones, ones, jnp.zeros(()))

@dataclass
class PCANormalizer:
    mean: Any
    S: Any
    U: Any

    @property
    def structure(self) -> T:
        return jax.ShapeDtypeStruct((self.S.shape[0],), jnp.float32)

    def normalize(self, x):
        centered = tree.map(lambda x, m: x - m, x, self.mean)
        centered_flat, _ = jax.flatten_util.ravel_pytree(centered)
        whitened = (self.U.T @ centered_flat) / self.S
        return whitened
    
    def unnormalize(self, x):
        centered = self.U @ (self.S * x)
        mean_flat, uf = jax.flatten_util.ravel_pytree(self.mean)
        unnormalized = uf(centered + mean_flat)
        return unnormalized

    @staticmethod
    def from_data(data, dims=None):
        data = PyTreeData.from_data(data).tree
        mean = tree.map(lambda x: jnp.mean(x, axis=0), data)
        data_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(data)
        C = jnp.cov(data_flat, rowvar=False)
        U, S, _ = jnp.linalg.svd(C)
        if dims is not None:
            U = U[:,:dims]
            S = S[:dims]
        return PCANormalizer(mean, S, U)