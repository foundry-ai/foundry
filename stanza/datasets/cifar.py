from pathlib import Path
import tarfile
import jax.numpy as jnp
import pickle
from stanza.data import Data
from stanza.datasets import builder
from .util import download_and_extract, cache_path

def _read_batch(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    data = jnp.array(dict[b"data"])
    labels = jnp.array(dict[b"labels"])
    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    return data, labels

@builder
def cifar10(quiet=False, splits=set()):
    tar_path = cache_path("cifar10", "cifar10.tar.gz")
    data_path = cache_path("cifar10", "data")
    if not data_path.exists():
        download_and_extract(tar_path, data_path, 
            url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            quiet=quiet
        )
    data_path = data_path / "cifar-10-batches-py"
    test_path = data_path / "test_batch"

    data = {}
    if "train" in splits:
        train_paths = [data_path / f"data_batch_{i}" for i in range(1,6)]
        train_batches = [_read_batch(p) for p in train_paths]
        train_data = jnp.concatenate([x[0] for x in train_batches])
        train_labels = jnp.concatenate([x[1] for x in train_batches])
        train = Data.from_pytree((train_data, train_labels))
        data["train"] = train
    if "test" in splits:
        test_data, test_labels = _read_batch(test_path)
        test = Data.from_pytree((test_data, test_labels))
        data["test"] = test
    return data

@builder
def cifar100(quiet=False, splits=set()):
    tar_path = Path(".cache/data/cifar100.tar.gz")
    dest_path = Path(".cache/data/cifar100")
    _download("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", tar_path,
              quiet=quiet)
    _extract(tar_path, dest_path)
    test_data = jnp.load(dest_path / "cifar-100-python" / "test", allow_pickle=True)
    print(test_data)