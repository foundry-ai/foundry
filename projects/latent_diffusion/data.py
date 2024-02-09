from stanza.dataclasses import dataclass
from stanza.data.normalizer import (
    Normalizer, LinearNormalizer, DummyNormalizer
)
from stanza.data import Data, PyTreeData

import stanza.datasets as datasets

import jax.numpy as jnp
import jax
from jax.random import PRNGKey

import stanza.graphics.canvas as canvas

from stanza.reporting import Image
from typing import Callable

@dataclass
class DataInfo:
    train_data: Data
    test_data: Data
    normalizer: Normalizer
    samples_visualizer: Callable
    n_visualize_samples: int

def generate_swissroll(rng_key, num_samples):
    thetas = jax.random.uniform(rng_key, (num_samples,),
                            minval=0, maxval=10)
    x, y = jnp.cos(thetas), jnp.sin(thetas)
    r = 0.9*thetas/10 + 0.1
    x, y = (r * x), (r * y)
    data = jnp.stack([x, y], axis=-1)
    return Data.from_pytree(data)

def generate_normal(rng_key, num_samples):
    dim = 8
    xs = jax.random.normal(rng_key, (num_samples, dim))
    L = jax.random.normal(PRNGKey(42), (xs.shape[-1],xs.shape[-1]))
    sigma = L.T @ L
    xs = sigma * xs
    return Data.from_pytree(xs)

def load_data(name, rng_key, train_samples=None, test_samples=None):
    if name == "swissroll":
        train_samples = train_samples or 10_000
        test_samples = test_samples or 100
        test_rng, train_rng = jax.random.split(rng_key)

        train = generate_swissroll(train_rng, train_samples)
        test = generate_swissroll(test_rng, test_samples)
        normalizer = LinearNormalizer.from_data(train)

        # draw the swissroll as dots
        def visualize_samples(samples, latents=None):
            img = jnp.ones((256, 256, 4))
            r = 0.05*jnp.ones((samples.shape[0],))
            if latents is None:
                circles = canvas.union_batch(canvas.circle(4*samples, r))
                sdf = canvas.fill(circles, color=(1, 0, 0, 0.5))
            else:
                s = (latents - jnp.min(latents))/(jnp.max(latents) - jnp.min(latents))
                s = jnp.squeeze(s, -1)
                z = jnp.zeros_like(s)
                colors = jnp.array([1*s, 1*(1-s), z])
                colors = jnp.moveaxis(colors, 0, -1)
                circles = canvas.fill(canvas.circle(4*samples, r), color=colors)
                sdf = canvas.stack_batch(circles)
            sdf = canvas.transform(sdf,
                translation=(5, 5),
                scale=img.shape[0]/10
            )
            img = canvas.paint(img, sdf)
            return {"samples": Image(img)}
        n_visualize_samples = 128
    elif name == "normal":
        train_samples = train_samples or 10_000
        test_samples = test_samples or 100
        test_rng, train_rng = jax.random.split(rng_key)

        train = generate_normal(train_rng, train_samples)
        test = generate_normal(test_rng, test_samples)
        normalizer = LinearNormalizer.from_data(train)

        def visualize_samples(samples):
            img = jnp.ones((256, 256, 4))
            r = 0.05*jnp.ones((samples.shape[0],))
            circles = canvas.batch_union(canvas.circle(4*samples, r))
            sdf = canvas.fill(circles, color=(1, 0, 0, 0.5))
            sdf = canvas.transform(sdf,
                translation=(5, 5),
                scale=img.shape[0]/10
            )
            img = canvas.paint(img, sdf)
            return {"samples": Image(img)}
        n_visualize_samples = 128
    else:
        train, test = datasets.load(
            name, splits=("train", "test"))
        def data_mapper(data):
            if name.startswith("cifar"):
                data = data.map(lambda x: x[0])
            def convert_images(x):
                # convert 0 to 255 to 0 to 1
                return (x.astype(jnp.float32) / 255.0)
            data = data.map(convert_images)
            return data
        train, test = (
            data_mapper(train),
            data_mapper(test)
        )
        normalizer = LinearNormalizer.from_data(train)
        def visualize_samples(samples):
            img = (255.*samples).astype(jnp.uint8)
            return {"samples": Image(img)}
        n_visualize_samples = 16
    return DataInfo(
        train,
        test,
        normalizer,
        visualize_samples,
        n_visualize_samples
    )