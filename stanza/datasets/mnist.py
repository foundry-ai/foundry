from stanza.datasets import Dataset
import stanza.data as du
import stanza.util as util

from .util import cache_path, download

import gzip

import jax
import jax.numpy as jnp
import struct
import array
import wandb
import numpy as np

def visualize_mnist(signal_samples,
                    latent_samples=None,
                    signal_resample=None):
    if signal_resample is not None:
        signal_samples = jnp.concatenate(
            (signal_samples, signal_resample),
            axis=-2 # concat along cols
        )
    return wandb.Image(np.array(
        util.grid(signal_samples))
    )


def load_mnist(quiet=False, **kwargs):
    with jax.default_device(jax.devices("cpu")[0]):
        data_path = cache_path("mnist")
        """Download and parse the raw MNIST dataset."""
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        def parse_labels(filename):
            with gzip.open(filename, "rb") as fh:
                _ = struct.unpack(">II", fh.read(8))
                return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

        def parse_images(filename):
            with gzip.open(filename, "rb") as fh:
                _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                img = jnp.array(array.array("B", fh.read()),
                        dtype=jnp.uint8).reshape(num_data, rows, cols)
                # Add channel dimension
                # convert to float32 and -1 to 1 range
                img = (img.astype(jnp.float32) / 255. - 0.5)*2
                return jnp.expand_dims(img, -1)

        for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
            download(data_path/filename, base_url + filename, quiet=quiet)
        
        train_images = parse_images(data_path / "train-images-idx3-ubyte.gz")
        train_labels = parse_labels(data_path / "train-labels-idx1-ubyte.gz")
        train_data = (train_images, train_labels)
        test_images = parse_images(data_path / "t10k-images-idx3-ubyte.gz")
        test_labels = parse_labels(data_path / "t10k-labels-idx1-ubyte.gz")
        test_data = (test_images, test_labels)
        return Dataset(
            splits={
                "train": du.PyTreeData(train_data),
                "test": du.PyTreeData(test_data)
            },
        )

DATASET_LOADERS = {"mnist":load_mnist}