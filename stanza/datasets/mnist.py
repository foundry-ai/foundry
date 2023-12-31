from stanza.datasets import builder
from stanza.data import Data

from .util import download, cache_path
from pathlib import Path as Path

import struct
import array
import gzip

import jax.numpy as jnp



@builder
def mnist(quiet=False, splits=set()):
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
            return jnp.array(array.array("B", fh.read()),
                      dtype=jnp.uint8).reshape(num_data, rows, cols)

    for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        download(data_path/filename, base_url + filename, quiet=quiet)
    
    data = {}
    if "train" in splits:
        train_images = parse_images(data_path / "train-images-idx3-ubyte.gz")
        train_labels = parse_labels(data_path / "train-labels-idx1-ubyte.gz")
        data["train"] = Data.from_pytree((train_images, train_labels))
    if "test" in splits:
        test_images = parse_images(data_path / "t10k-images-idx3-ubyte.gz")
        test_labels = parse_labels(data_path / "t10k-labels-idx1-ubyte.gz")
        data["test"] = Data.from_pytree((test_images, test_labels))
    return data