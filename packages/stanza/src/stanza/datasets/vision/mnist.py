from stanza.datasets import DatasetRegistry
import stanza.data as du
from stanza.data import normalizer as nu
from stanza.data.transform import (
    random_horizontal_flip, random_subcrop, random_cutout
)

from ..util import cache_path, download

from . import ImageClassDataset, LabeledImage

import gzip

import jax
import jax.numpy as jnp
import struct
import array
import wandb
import numpy as np

def _standard_augmentations(rng_key, x):
    x = random_subcrop(rng_key, x, (28, 28), 4)
    return x

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

        for job_name, filename, md5 in [
                    ("MNIST Train Images", "train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                    ("MNIST Train Labels", "train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
                    ("MNIST Test Images", "t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
                    ("MNIST Test Labels", "t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
                ]:
            download(data_path/filename, url=base_url + filename, 
                    job_name=job_name, md5=md5,
                    quiet=quiet)
        
        train_images = parse_images(data_path / "train-images-idx3-ubyte.gz")
        train_labels = parse_labels(data_path / "train-labels-idx1-ubyte.gz")
        train_data = LabeledImage(train_images, train_labels)
        test_images = parse_images(data_path / "t10k-images-idx3-ubyte.gz")
        test_labels = parse_labels(data_path / "t10k-labels-idx1-ubyte.gz")
        test_data = LabeledImage(test_images, test_labels)

        train_normalized = du.PyTreeData(LabeledImage((train_images.astype(jnp.float32) / 128.0) - 1, None))
        return ImageClassDataset(
            splits={
                "train": du.PyTreeData(train_data),
                "test": du.PyTreeData(test_data)
            },
            normalizers={
                "pixel_standard_dev": lambda: (nu.Compose(
                    (nu.Chain([
                        nu.ImageNormalizer(jax.ShapeDtypeStruct((28, 28, 1), jnp.uint8)),
                        nu.StdNormalizer.from_data(train_normalized, component_wise=True)
                    ]), 
                     nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
                )),
                "standard_dev": lambda: (nu.Compose(
                    (nu.Chain([
                        nu.ImageNormalizer(jax.ShapeDtypeStruct((28, 28, 1), jnp.uint8)),
                        nu.StdNormalizer.from_data(train_normalized, component_wise=False)
                    ]), 
                     nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
                )),
                "pca": lambda dims=None: (nu.Compose(
                    (nu.Chain([
                        nu.ImageNormalizer(jax.ShapeDtypeStruct((28, 28, 1), jnp.uint8)),
                        # transform to gaussian
                        nu.PCANormalizer.from_data(train_normalized, dims=dims)
                    ]), 
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
                )),
                "hypercube": lambda: (nu.Compose(
                    (nu.ImageNormalizer(jax.ShapeDtypeStruct((28, 28, 1), jnp.uint8)), 
                     nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
                ))
            },
            transforms={
                "standard_augmentations": lambda: _standard_augmentations
            },
            classes=[str(i) for i in range(10)]
        )

datasets = DatasetRegistry[ImageClassDataset]()
datasets.register(_load_mnist)