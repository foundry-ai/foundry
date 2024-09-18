import jax
import foundry.numpy as jnp

import pickle
from foundry.data import PyTreeData
from foundry.data import normalizer as nu
from foundry.datasets import DatasetRegistry
from . import ImageClassDataset, LabeledImage
from ..util import download, extract, cache_path
from foundry.data.transform import (
    random_horizontal_flip, random_subcrop, random_cutout
)

def _read_batch(file):
    dict = pickle.load(file, encoding="bytes")
    data = jnp.array(dict[b"data"], dtype=jnp.uint8)
    labels = jnp.array(dict[b"labels"] if b"labels" in dict else dict[b"fine_labels"], dtype=jnp.uint8)
    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    return data, labels

def _standard_augmentations(rng_key, x):
    a, b, c = jax.random.split(rng_key, 3)
    x = random_horizontal_flip(a, x)
    #x = random_subcrop(b, x, (32, 32), 4, padding_mode="edge")
    # x = random_cutout(c, x, 8, 0.1)
    return x

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
    train_normalized = PyTreeData((train_data.astype(jnp.float32) / 128.0) - 1)
    return ImageClassDataset(
        splits=data,
        normalizers={
            "hypercube": lambda: nu.Compose(
                LabeledImage(
                    nu.ImageNormalizer(jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8)),
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8))
                )
            ),
            "standard_dev": lambda: nu.Compose(
                LabeledImage(
                    nu.Chain([nu.ImageNormalizer(jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8)),
                        nu.StdNormalizer.from_data(train_normalized, component_wise=False)]),
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8))
                )
            )
        },
        transforms={
            "standard_augmentations": lambda: _standard_augmentations
        },
        classes=["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
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
    return ImageClassDataset(
        splits=data,
        normalizers={
            "hypercube": lambda: nu.Compose(
                (nu.ImageNormalizer(LabeledImage(jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8), None)), 
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
            )
        },
        transforms={
            "standard_augmentations": lambda: _standard_augmentations
        },
        classes=classes
    )

datasets = DatasetRegistry[ImageClassDataset]()
datasets.register("cifar10", _load_cifar10)
datasets.register("cifar100", _load_cifar100)
