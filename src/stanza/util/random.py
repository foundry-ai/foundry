import jax.random
import jax.numpy as jnp

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
        k, n = jax.jit(
            jax.random.split,
            static_argnums=1
        )(self._key)
        self._key = k
        return n