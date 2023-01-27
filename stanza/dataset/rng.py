from stanza.dataset import Dataset, INFINITE

import jax.random

class RNGDataset(Dataset):
    def __init__(self, rng_key):
        self.rng_key = rng_key

    @property
    def start(self):
        return self.rng_key

    def remaining(self, iterator):
        return INFINITE
    
    def next(self, iterator):
        rng, _ = jax.random.split(iterator)
        return rng
    
    def get(self, iterator):
        _, sk = jax.random.split(iterator)
        return sk