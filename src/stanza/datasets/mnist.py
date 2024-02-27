from stanza.datasets import ImageClassDataset, DatasetRegistry
import stanza.data as du
from stanza.data import normalizer as nu

from .util import cache_path, download

import gzip

import jax
import jax.numpy as jnp
import struct
import array
import wandb
import numpy as np

def _load_mnist(quiet=False, **kwargs):
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
                return jnp.expand_dims(img, -1)

        for job_name, filename in [
                    ("MNIST Train Images", "train-images-idx3-ubyte.gz"),
                    ("MNIST Train Labels", "train-labels-idx1-ubyte.gz"),
                    ("MNIST Test Images", "t10k-images-idx3-ubyte.gz"),
                    ("MNIST Test Labels", "t10k-labels-idx1-ubyte.gz")
                ]:
            download(data_path/filename, url=base_url + filename, 
                    job_name=job_name,
                    quiet=quiet)
        
        train_images = parse_images(data_path / "train-images-idx3-ubyte.gz")
        train_labels = parse_labels(data_path / "train-labels-idx1-ubyte.gz")
        train_data = (train_images, train_labels)
        test_images = parse_images(data_path / "t10k-images-idx3-ubyte.gz")
        test_labels = parse_labels(data_path / "t10k-labels-idx1-ubyte.gz")
        test_data = (test_images, test_labels)
        return ImageClassDataset(
            splits={
                "train": du.PyTreeData(train_data),
                "test": du.PyTreeData(test_data)
            },
            normalizers={
                "hypercube": nu.Compose(
                    (nu.ImageNormalizer(jax.ShapeDtypeStruct((28, 28, 1), jnp.uint8)), 
                     nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
                )
            },
            classes=[str(i) for i in range(10)]
        )

registry = DatasetRegistry[ImageClassDataset]()
registry.register("mnist", _load_mnist)