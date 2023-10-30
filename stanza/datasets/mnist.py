from stanza.datasets import builder
from stanza.data import Data

from .util import download as _download
from pathlib import Path as Path

import struct
import array
import gzip

import jax.numpy as jnp



@builder
def mnist(quiet=False, splits=set()):
    _DATA = Path(".cache/data/mnist")
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
        _download(base_url + filename, _DATA / filename, quiet=quiet)
    
    if "train" in splits:
        train_images = parse_images(_DATA / "train-images-idx3-ubyte.gz")
        train_labels = parse_labels(_DATA / "train-labels-idx1-ubyte.gz")
    test_images = parse_images(_DATA / "t10k-images-idx3-ubyte.gz")
    test_labels = parse_labels(_DATA / "t10k-labels-idx1-ubyte.gz")
    train = Data.from_pytree((train_images, train_labels))
    test = Data.from_pytree((test_images, test_labels))
    return train, test