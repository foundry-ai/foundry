from stanza.dataset import Dataset, INFINITE
from stanza.util.dataclasses import dataclass

import jax.random

@dataclass(jax=True)
class RNGDataset(Dataset):
    rng_key: jax.random.PRNGKey

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