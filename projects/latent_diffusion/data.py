from stanza.dataclasses import dataclass
from stanza.data import Data

import stanza.datasets as datasets

import jax.numpy as jnp
import jax

import stanza.graphics.canvas as canvas

from stanza.reporting import Image
from typing import Callable

@dataclass(jax=True)
class DataInfo:
    train_data: Data
    test_data: Data
    samples_visualizer: Callable

def generate_swissroll(rng_key, num_samples):
    thetas = jax.random.uniform(rng_key, (num_samples,),
                            minval=0, maxval=10)
    x, y = jnp.cos(thetas), jnp.sin(thetas)
    r = thetas
    x, y = (r * x), (r * y)
    data = jnp.stack([x, y], axis=-1)
    return Data.from_pytree(data)

def load_data(name, rng_key, train_samples=None, test_samples=None):
    if name == "swissroll":
        train_samples = train_samples or 10_000
        test_samples = test_samples or 100
        test_rng, train_rng = jax.random.split(rng_key)

        train = generate_swissroll(train_rng, train_samples)
        test = generate_swissroll(test_rng, test_samples)

        # draw the swissroll as dots
        def visualize_samples(samples):
            img = jnp.ones((256, 256, 4))
            r = 0.01*jnp.ones((samples.shape[0],))
            circles = canvas.batch_union(canvas.circle(samples, r))
            sdf = canvas.fill(circles, color=(1, 0, 0, 0.5))
            sdf = canvas.transform(sdf,
                translation=(5, 5),
                scale=img.shape[0]/10
            )
            img = canvas.paint(img, sdf)
            return img
    else:
        train, test = datasets.load(
            name, splits=("train", "test"))
        def data_mapper(data):
            if name.startswith("cifar"):
                data = data.map(lambda x: x[0])
            def convert_images(x):
                # convert 0 to 255 to -1 to 1
                return 2*(x.astype(jnp.float32) / 255) - 1.
            data = data.map(convert_images)
            return data
        train, test = (
            data_mapper(train),
            data_mapper(test)
        )
        def visualize_samples(samples):
            return {"samples": Image(samples)}
    return DataInfo(
        train,
        test,
        visualize_samples
    )