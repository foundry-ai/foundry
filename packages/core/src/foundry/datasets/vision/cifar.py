import foundry.random
import foundry.numpy as jnp

import pickle

from foundry.data import PyTreeData, Data
from foundry.data import normalizer as nu
from foundry.core.dataclasses import dataclass
from foundry.core import ShapeDtypeStruct

from . import ImageClassDataset, LabeledImage
from ..util import download, extract, cache_path

from foundry.data.transform import (
    Transform,
    random_horizontal_flip, random_subcrop, random_cutout
)

def _read_batch(file):
    dict = pickle.load(file, encoding="bytes")
    data = jnp.array(dict[b"data"], dtype=jnp.uint8)
    labels = jnp.array(dict[b"labels"] if b"labels" in dict else dict[b"fine_labels"], dtype=jnp.uint8)
    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    return data, labels

@dataclass
class CIFARDataset(ImageClassDataset):
    _splits : dict[str, PyTreeData[LabeledImage]]
    _classes : list[str]
    _mean : jnp.ndarray
    _std : jnp.ndarray

    @property
    def classes(self) -> list[str]:
        return self._classes

    def split(self, name) -> Data[LabeledImage]:
        return self._splits[name]
    
    def augmentations(self, name) -> Transform[LabeledImage, LabeledImage]:
        if name == "classifier":
            def classifier_augmentation(rng_key, sample : LabeledImage) -> LabeledImage:
                pixels = sample.pixels
                pixels = random_horizontal_flip(rng_key, pixels)
                pixels = random_subcrop(rng_key, pixels, (32, 32))
                pixels = random_cutout(rng_key, pixels, (8, 8))
                return LabeledImage(pixels, sample.label)
            return classifier_augmentation
        elif name == "generator":
            def generator_augmentation(rng_key, sample : LabeledImage) -> LabeledImage:
                pixels = sample.pixels
                pixels = random_horizontal_flip(rng_key, pixels)
                return LabeledImage(pixels, sample.label)
            return generator_augmentation
        return None
    
    def normalizer(self, name) -> nu.Normalizer[LabeledImage]:
        if name == "hypercube":
            return nu.Compose(
                LabeledImage(
                    pixels=nu.ImageNormalizer(ShapeDtypeStruct((32,32,3), jnp.uint8)),
                    label=nu.DummyNormalizer(ShapeDtypeStruct((), jnp.uint8))
                )
            )
        elif name == "standard_dev":
            return nu.Compose(LabeledImage(
                    pixels=nu.Chain([
                        nu.ImageNormalizer(ShapeDtypeStruct((32,32,3), jnp.uint8)),
                        nu.StdNormalizer(
                            mean=self._mean,
                            std=self._std,
                            var=jnp.square(self._std)
                        )
                    ]),
                    label=nu.DummyNormalizer(ShapeDtypeStruct((), jnp.uint8))
                ))
        elif name == "pixel_standard_dev":
            return nu.Compose(LabeledImage(
                    pixels=nu.Chain([
                        nu.ImageNormalizer(ShapeDtypeStruct((32,32,3), jnp.uint8)),
                        nu.StdNormalizer(
                            mean=jnp.mean(self._mean),
                            std=jnp.sqrt(jnp.sum(jnp.square(self._std))),
                            var=jnp.sum(jnp.square(self._std))
                        )
                    ]),
                    label=nu.DummyNormalizer(ShapeDtypeStruct((), jnp.uint8))
                ))
        return None

def _load_cifar10(quiet=False):
    tar_path = cache_path("cifar10", "cifar10.tar.gz")
    download(tar_path,
        job_name="CIFAR-10",
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a",
        quiet=quiet
    )
    train_batches = [None]*5
    test_data = None
    test_labels = None
    def extract_handler(info, file):
        nonlocal test_data, test_labels
        if info.is_dir:
            return
        if info.filename.startswith("cifar-10-batches-py/data_batch_"):
            batch_num = int(info.filename[-1])
            train_batches[batch_num-1] = _read_batch(file)
        elif info.filename == "cifar-10-batches-py/test_batch":
            test_data, test_labels = _read_batch(file)
    extract(tar_path, extract_handler,
            job_name="CIFAR-10", quiet=True)
    train_data = jnp.concatenate([x[0] for x in train_batches])
    train_labels = jnp.concatenate([x[1] for x in train_batches])

    train = PyTreeData(LabeledImage(train_data, train_labels))
    test = PyTreeData(LabeledImage(test_data, test_labels))
    data = {
        "train": train,
        "test": test
    }
    return CIFARDataset(
        _splits=data,
        _classes=["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"],
        _mean=jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32),
        _std=jnp.array([0.2023, 0.1994, 0.2010], dtype=jnp.float32)
    )

def _load_cifar100(quiet=False):
    tar_path = cache_path("cifar100", "cifar100.tar.gz")
    download(tar_path,
        job_name="CIFAR-100",
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
        quiet=quiet
    )
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    classes = None
    def extract_handler(info, file):
        nonlocal train_data, train_labels
        nonlocal test_data, test_labels
        nonlocal classes
        if info.is_dir:
            return
        if info.filename == "cifar-100-python/train":
            train_data, train_labels = _read_batch(file)
        elif info.filename == "cifar-100-python/test":
            test_data, test_labels = _read_batch(file)
        elif info.filename == "cifar-100-python/meta":
            meta = pickle.load(file, encoding="bytes")
            fine_label_names = [x.decode("utf-8") for x in meta[b"fine_label_names"]]
            classes = fine_label_names
    extract(tar_path, extract_handler,
            job_name="CIFAR-100", quiet=True)
    train = PyTreeData(LabeledImage(train_data, train_labels))
    test = PyTreeData(LabeledImage(test_data, test_labels))
    data = {
        "train": train,
        "test": test
    }
    return CIFARDataset(
        _splits=data,
        _classes=classes,
        _mean=jnp.array([0.5071, 0.4867, 0.4408], dtype=jnp.float32),
        _std=jnp.array([0.2675, 0.2565, 0.2761], dtype=jnp.float32),
    )

from foundry.datasets.core import DatasetRegistry

def register(registry : DatasetRegistry, prefix=None):
    registry.register("cifar10", _load_cifar10, prefix=prefix)
    registry.register("cifar100", _load_cifar100, prefix=prefix)