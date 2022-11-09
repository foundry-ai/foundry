from .dataset import Dataset, INFINITE
from collections import NamedTuple

import jax.random
import jax.numpy as jnp
import numpy as np

def key_or_seed(key_or_seed):
    if isinstance(key_or_seed, int):
        key_or_seed = jax.random.PRNGKey(key_or_seed)
    elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
            hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
        key_or_seed = jax.random.PRNGKey(key_or_seed)

class PRNGSequence:
    def __init__(self, key_or_seed):
        self._key = jinx.random.key_or_seed(key_or_seed)
    
    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n

class PRNGDataset(Dataset):
    def __init__(self, key):
        self.key = key

    @property
    def length(self):
        return INFINITE
    
    @property
    def start(self):
        return jax.random.split(self.key)

    def is_end(self, iterator):
        return False

    def next(self, iterator):
        _, n = iterator
        h, n = jax.random.split(n)
        return h,n
    
    def get(self, iterator):
        h, _ = iterator
        return h
    

class PRNGKey(NamedTuple):
    idx: np.ndarray
    block: jax.random.PRNGKey
    next_block_key: jax.random.PRNGKey