from stanza.dataclasses import dataclass, field
from stanza.datasets import DatasetRegistry
from stanza.util.registry import from_module

from stanza.data import Data

from rich.progress import track

from . import NLPDataset
from .tokenizer import Tokenizer, iterate_raw

import stanza.datasets.util as du
import numpy as np
import os
import logging
import jax
import jax.numpy as jnp

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
                arrays.append(jnp.array(self.mmap[i:i+self.sample_size]))
            return jnp.stack(arrays)


    def __getitem__(self, i):
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
                           splits={"train", "test"}):
    data = {}
    def make_tokenizer(split="train"):
        tokenizer_path = du.cache_path("tiny_stories") / f"tokenizer.model"
        vocab_path = du.cache_path("tiny_stories") / f"tokenizer.vocab"
        if tokenizer_path.exists() and vocab_path.exists():
            return Tokenizer.load_model(tokenizer_path, vocab_path)
        if not quiet:
            logger.info("Training tokenizer...")
        data_path = du.cache_path("tiny_stories") / f"{split}.txt"
        if not data_path.exists():
            du.download(data_path,
                url=URL.format(split=split),
                quiet=quiet
            )
        tokenizer = Tokenizer.train(
            iterate_raw(data_path, "<|endoftext|>", "<|n|>", 2048),
            user_defined_symbols=["<|n|>"],
            vocab_size=1024
        )
        tokenizer.save_model(tokenizer_path, vocab_path)
        if not quiet:
            logger.info("Done training tokenizer.")
        return tokenizer
    tokenizer = make_tokenizer()

    def load_split(name):
        data_path = du.cache_path("tiny_stories") / f"{name}.txt"
        bin_path = du.cache_path("tiny_stories") / f"{name}.bin"
        if not data_path.exists():
            du.download(data_path,
                url=URL.format(split=name),
                quiet=quiet
            )
        if not bin_path.exists():

            with open(bin_path, "wb") as f:
                iterator = iterate_raw(data_path,
                    "<|endoftext|>", "<|n|>", 2048
                )
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
    if "train" in splits:
        data["train"] = load_split("train")
    if "test" in splits:
        data["test"] = load_split("valid")
    return tokenizer, data

def load_tiny_stories(*,quiet=False):
    tokenizer, splits = load_tiny_stories_data(quiet=quiet)

    return NLPDataset(
        splits=splits,
        tokenizer=tokenizer
    )

datasets : DatasetRegistry[TinyStoriesData] = DatasetRegistry()
datasets.register(load_tiny_stories)