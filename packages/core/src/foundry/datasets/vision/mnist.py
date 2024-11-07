from foundry.core import Array, ShapeDtypeStruct
from foundry.core.dataclasses import dataclass
from foundry.datasets.core import DatasetRegistry
from foundry.data import Data, PyTreeData
from ..util import cache_path, download
from . import ImageClassDataset, LabeledImage

from functools import partial

import foundry.data.normalizer as normalizers
import foundry.numpy as jnp

import gzip
import jax
import struct
import array

@dataclass
class MnistDataset(ImageClassDataset):
    _splits : dict[str, Data[LabeledImage]]
    _classes : list[str]
    _mean: Array
    _std: Array
    _var: Array

    @property
    def classes(self) -> list[str]:
        return self._classes
    
    def split(self, name) -> Data[LabeledImage]:
        return self._splits[name]
    
    def augmentation(self, name):
        import foundry.data.transform as transforms
        if name == "classifier":
            def classifier_augmentation(rng_key, sample : LabeledImage) -> LabeledImage:
                pixels = sample.pixels
                pixels = transforms.random_horizontal_flip(rng_key, pixels)
                pixels = transforms.random_subcrop(rng_key, pixels, (32, 32))
                pixels = transforms.random_cutout(rng_key, pixels, (8, 8))
                return LabeledImage(pixels, sample.label)
            return classifier_augmentation
        elif name == "generator":
            def generator_augmentation(rng_key, sample : LabeledImage) -> LabeledImage:
                pixels = sample.pixels
                pixels = transforms.random_horizontal_flip(rng_key, pixels)
                return LabeledImage(pixels, sample.label)
            return generator_augmentation

    def normalizer(self, name) -> normalizers.Normalizer[LabeledImage]:
        if name == "hypercube":
            return normalizers.Compose(
                LabeledImage(
                    pixels=normalizers.ImageNormalizer(ShapeDtypeStruct((28,28,3), jnp.uint8)),
                    label=normalizers.Identity(ShapeDtypeStruct((), jnp.uint8))
                )
            )
        elif name == "standard_dev":
            return normalizers.Compose(LabeledImage(
                    pixels=normalizers.Chain([
                        normalizers.ImageNormalizer(ShapeDtypeStruct((32,32,3), jnp.uint8)),
                        normalizers.StdNormalizer(
                            mean=self._mean,
                            std=self._std,
                            var=jnp.square(self._std)
                        )
                    ]),
                    label=normalizers.Identity(ShapeDtypeStruct((), jnp.uint8))
                ))
        elif name == "pixel_standard_dev":
            return normalizers.Compose(LabeledImage(
                    pixels=normalizers.Chain([
                        normalizers.ImageNormalizer(ShapeDtypeStruct((32,32,3), jnp.uint8)),
                        normalizers.StdNormalizer(
                            mean=jnp.mean(self._mean),
                            std=jnp.sqrt(jnp.sum(jnp.square(self._std))),
                            var=jnp.sum(jnp.square(self._std))
                        )
                    ]),
                    label=normalizers.Identity(ShapeDtypeStruct((), jnp.uint8))
                ))
        return None


def _load_mnist(quiet=False, classes=None, **kwargs):
    classes = jnp.array(classes) if classes is not None else None
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
                _, normalizersm_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                img = jnp.array(array.array("B", fh.read()),
                        dtype=jnp.uint8).reshape(normalizersm_data, rows, cols)
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
        train_data = LabeledImage(
            parse_images(data_path / "train-images-idx3-ubyte.gz"),
            parse_labels(data_path / "train-labels-idx1-ubyte.gz")
        )
        test_data = LabeledImage(
            parse_images(data_path / "t10k-images-idx3-ubyte.gz"),
            parse_labels(data_path / "t10k-labels-idx1-ubyte.gz")
        )

        def filter(data : LabeledImage):
            if classes is None:
                return data
            else:
                mask = jax.vmap(lambda s: jnp.any(s.label == classes))(data)
                filtered = LabeledImage(data.pixels[mask], data.label[mask])
                # reindex the labels using the classes array
                new_labels = jax.vmap(lambda l: jnp.argmax(classes == l))(filtered.label)
                return LabeledImage(filtered.pixels, new_labels)

        train_data = filter(train_data)
        test_data = filter(test_data)

        train_norm_images = (train_data.pixels.astype(jnp.float32) / 128.0) - 1

        return MnistDataset(
            _splits={
                "train": PyTreeData(train_data),
                "test": PyTreeData(test_data)
            },
            _classes=[str(i) for i in range(10)] if classes is None else [str(i) for i in classes],
            _mean=jnp.mean(train_norm_images, axis=(0,)),
            _std=jnp.std(train_norm_images, axis=(0,)),
            _var=jnp.var(train_norm_images, axis=(0,))
        )

    
def register(registry : DatasetRegistry, prefix=None):
    registry.register("mnist", _load_mnist, prefix=prefix)
    registry.register("mnist/binary/0_9", partial(_load_mnist, classes=[0,9]), prefix=prefix)