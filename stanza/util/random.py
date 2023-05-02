from stanza.data import Data, INFINITE
from functools import partial
from typing import NamedTuple

import jax.random
import jax.numpy as jnp
import numpy as np

def key_or_seed(key_or_seed):
    if isinstance(key_or_seed, int):
        key_or_seed = jax.random.PRNGKey(key_or_seed)
    elif (hasattr(key_or_seed, "shape") and
            hasattr(key_or_seed, "dtype") and \
            key_or_seed.dtype == jnp.uint32):
        key_or_seed = key_or_seed
    else:
        raise ValueError("Not key or seed!")
    return key_or_seed

class PRNGSequence:
    def __init__(self, key_or_val):
        self._key = key_or_seed(key_or_val)
    
    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n

class PRNGDataset(Data):
    def __init__(self, key):
        self.key = key

    def is_end(self, iterator):
        return False

    def remaining(self, iterator):
        return INFINITE

    @property
    def start(self):
        return jax.random.split(self.key)

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

# A modified version of the jax permutation implementation
# that will only permute the first n elements
from jax._src.random import _random_bits, _split, _check_prng_key

def permutation(key, dim, n):
    key, _ = _check_prng_key(key)
    return _permutation(key, dim, n)

@partial(jax.jit, static_argnums=(1,), inline=True)
def _permutation(key, dim, n) -> jax.Array:
    exponent = 3
    uint32max = jnp.iinfo(np.uint32).max
    num_rounds = int(np.ceil(exponent * np.log(max(1, dim)) / np.log(uint32max)))
    x = jnp.arange(dim)
    # fill only the first n
    mask = x < n
    for _ in range(num_rounds):
        key, subkey = _split(key)
        sort_keys = _random_bits(subkey, 32, x.shape)
        sort_keys = jnp.where(mask, sort_keys, jnp.array(uint32max,dtype=jnp.uint32))
        _, x = jax.lax.sort_key_val(sort_keys, x, 0)
    return x