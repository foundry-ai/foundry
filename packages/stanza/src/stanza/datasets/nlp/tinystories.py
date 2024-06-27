from stanza.dataclasses import dataclass
from stanza.datasets import DatasetRegistry, Dataset
from stanza.util.registry import from_module

from stanza.data.io import jax_static_cache

import stanza.datasets.util as du

@dataclass
class TinyStoriesData(Data):
    path : str = field(pytree_node=False)

    def __getitem__(self, i):
        pass

    # Get the amount of data:
    @staticmethod
    def _num_samples(path):
        return os.stat(path).st_size

    def __len__(self) -> int:
        return jax_static_cache(self._num_samples)(self.path)

def load_tiny_stories_data(*, quiet=False, 
            tokenizer = None, splits={"train", "test"}):
    data = {}
    def load_split(name):
        data_path = du.cache_path("tiny_stories") / f"{name}.txt"
        bin_path = du.cache_path("tiny_stories") / f"{name}.bin"
        if not data_path.exists():
            du.download(data_path,
                url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
                quiet=quiet
            )
        if name == "train" and not tokenizer.exists():
            pass
        if not bin_path.exists():
            du.tokenize_file(tokenizer,
                data_path,
                bin_path
            )
        return TinyStoriesData(bin_path)
    if "train" in splits:
        data["train"] = load_split()
    if "valid" in splits:
        load_split("valid")
    return tokenizer, data

def load_tiny_stories(*,quiet=False, tokenizer=None):
    tokenizer, splits = load_tiny_stories_data(tokenizer=tokenizer, quiet=quiet)

    return NLPDataset(
        splits=splits,
        tokenizer=tokenizer
    )

datasets : DatasetRegistry[TinyStoriesData] = DatasetRegistry()
datasets.register(load_tiny_stories)