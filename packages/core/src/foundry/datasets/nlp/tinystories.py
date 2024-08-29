from foundry.core.dataclasses import dataclass, field
from foundry.datasets import DatasetRegistry
from foundry.util.registry import from_module

from foundry.data import Data

from rich.progress import track

from . import NLPDataset
from .tokenizer import Tokenizer, iterate_raw, filter_ascii

import foundry.datasets.util as du
import numpy as np
import os
import logging
import jax
import foundry.numpy as jnp

logger = logging.getLogger(__name__)

@dataclass
class TinyStoriesData(Data):
    mmap : np.array = field(pytree_node=False)
    offset : int = field(pytree_node=False)
    length : int = field(pytree_node=False)
    sample_size : int = field(pytree_node=False)

    def _get(self, i):
        if i.shape == ():
            return self.mmap[i:i+self.sample_size]
        else:
            arrays = []
            for i in i:
                arrays.append(jnp.array(self.mmap[i:i+self.sample_size], dtype=jnp.uint16))
            return jnp.stack(arrays)


    def __getitem__(self, i):
        i = jnp.array(i, dtype=jnp.uint64)
        assert i.shape == (), i.dtype == jnp.uint64
        return jax.pure_callback(self._get,
                jax.ShapeDtypeStruct((self.sample_size,), np.uint16),
                i,
                vectorized=True
            )

    def __len__(self) -> int:
        return self.length - self.sample_size
    
    @property
    def structure(self):
        return jax.ShapeDtypeStruct((self.sample_size,), np.uint16)


URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"

def load_tiny_stories_data(*, quiet=False, sample_size=1024,
                            download_only=False,
                           splits={"train", "test"}):
    def download_split(split="train"):
        if split == "test": split = "valid"
        data_path = du.cache_path("tiny_stories") / f"{split}.txt"
        if not data_path.exists():
            du.download(data_path,
                url=URL.format(split=split),
                quiet=quiet
            )
        return data_path
    for split in splits:
        download_split(split)
    if download_only:
        return

    def make_tokenizer(split="train"):
        data_path = download_split(split)
        tokenizer_path = du.cache_path("tiny_stories") / f"tokenizer.model"
        vocab_path = du.cache_path("tiny_stories") / f"tokenizer.vocab"
        if tokenizer_path.exists() and vocab_path.exists():
            return Tokenizer.load_model(tokenizer_path)
        if not quiet:
            logger.info("Training tokenizer...")
        tokenizer = Tokenizer.train(
            filter_ascii(iterate_raw(
                data_path, "<|endoftext|>", 
                "<|n|>", 2048
            )),
            user_defined_symbols=["<|n|>"],
            vocab_size=1024
        )
        tokenizer.save_model(tokenizer_path, vocab_path)
        if not quiet:
            logger.info("Done training tokenizer.")
        return tokenizer
    tokenizer = make_tokenizer()

    def load_split(name):
        data_path = download_split(name)
        bin_path = du.cache_path("tiny_stories") / f"{name}.bin"
        if not data_path.exists():
            du.download(data_path,
                url=URL.format(split=name),
                quiet=quiet
            )
        if not bin_path.exists():
            with open(bin_path, "wb") as f:
                iterator = filter_ascii(iterate_raw(data_path,
                    "<|endoftext|>", "<|n|>", 2048
                ))
                if not quiet:
                    iterator = track(iterator, description=f"Encoding...", total=None)
                tokenizer.encode_to_file(
                    iterator, f
                )
        # data is stored as uint8's
        length = os.stat(bin_path).st_size // 2
        f = open(bin_path, "r+b")
        mm = np.memmap(
            bin_path, dtype=np.uint16, 
            mode="r", offset=0, shape=(length,))
        return TinyStoriesData(mm, 0, length, sample_size)

    data = dict({k: load_split(k) for k in splits})
    return tokenizer, data

def load_tiny_stories(*,quiet=False, download_only=False, **kwargs):
    tokenizer, splits = load_tiny_stories_data(quiet=quiet, download_only=download_only)

    return NLPDataset(
        splits=splits,
        tokenizer=tokenizer
    )

datasets : DatasetRegistry[TinyStoriesData] = DatasetRegistry()
datasets.register(load_tiny_stories)