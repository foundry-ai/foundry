import jax
import jax.numpy as jnp

import pickle
from stanza.data import PyTreeData
from stanza.data import normalizer as nu
from stanza.datasets import DatasetRegistry, ImageClassDataset
from .util import download, extract, cache_path

def _read_batch(file):
    dict = pickle.load(file, encoding="bytes")
    data = jnp.array(dict[b"data"])
    labels = jnp.array(dict[b"labels"] if b"labels" in dict else dict[b"fine_labels"])
    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    return data, labels

def _load_cifar10(quiet=False):
    tar_path = cache_path("cifar10", "cifar10.tar.gz")
    download(tar_path,
        job_name="CIFAFR-10",
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

    train = PyTreeData((train_data, train_labels))
    test = PyTreeData((test_data, test_labels))
    data = {
        "train": train,
        "test": test
    }
    return ImageClassDataset(
        splits=data,
        normalizers={
            "hypercube": nu.Compose(
                (nu.ImageNormalizer(jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8)),
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
            )
        },
        classes=["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
    )

def _load_cifar100(quiet=False):
    tar_path = cache_path("cifar100", "cifar100.tar.gz")
    download(tar_path,
        job_name="CIFAFR-100",
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
    train = PyTreeData((train_data, train_labels))
    test = PyTreeData((test_data, test_labels))
    data = {
        "train": train,
        "test": test
    }
    return ImageClassDataset(
        splits=data,
        normalizers={
            "hypercube": nu.Compose(
                (nu.ImageNormalizer(jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8)), 
                    nu.DummyNormalizer(jax.ShapeDtypeStruct((), jnp.uint8)))
            )
        },
        classes=classes
    )

registry = DatasetRegistry[ImageClassDataset]()
registry.register("cifar10", _load_cifar10)
registry.register("cifar100", _load_cifar100)