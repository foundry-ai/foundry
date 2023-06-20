from pathlib import Path
import requests
import tarfile
import jax.numpy as jnp
import pickle
from rich.progress import Progress
from stanza.data import Data

def _download(url, path, quiet=False):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*10 #10 Kibibyte
    if quiet:
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
    else:
        with Progress() as pbar:
            task = pbar.add_task("Downloading...", total=total_size_in_bytes)
            with open(path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(task, advance=len(data))

def _extract(tar_path, dest_dir):
    if dest_dir.is_dir():
        return
    dest_dir.mkdir()
    with tarfile.open(tar_path) as f:
        f.extractall(dest_dir)

def _read_batch(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    data = jnp.array(dict[b"data"])
    labels = jnp.array(dict[b"labels"])
    data = data.reshape((-1, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    return data, labels

def cifar10(quiet=False):
    tar_path = Path(".cache/data/cifar10.tar.gz")
    dest_path = Path(".cache/data/cifar10")
    _download("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path,
              quiet=quiet)
    _extract(tar_path, dest_path)
    dest_path = dest_path / "cifar-10-batches-py"
    test_path = dest_path / "test_batch"
    train_paths = [dest_path / f"data_batch_{i}" for i in range(1,6)]
    train_batches = [_read_batch(p) for p in train_paths]
    train_data = jnp.concatenate([x[0] for x in train_batches])
    train_labels = jnp.concatenate([x[1] for x in train_batches])
    test_data, test_labels = _read_batch(test_path)
    train = Data.from_pytree((train_data, train_labels))
    test = Data.from_pytree((test_data, test_labels))
    return train, test

def cifar100(quiet=False):
    tar_path = Path(".cache/data/cifar100.tar.gz")
    dest_path = Path(".cache/data/cifar100")
    _download("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", tar_path,
              quiet=quiet)
    _extract(tar_path, dest_path)
    test_data = jnp.load(dest_path / "cifar-100-python" / "test", allow_pickle=True)
    print(test_data)