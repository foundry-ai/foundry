from stanza.data import Data
import jax.numpy as jnp

def test_simple():
    dataset = Data.from_pytree(jnp.arange(128))
    assert dataset.length == 128
    batched_dataset = dataset.batch(32)
    assert batched_dataset.data.data.shape == (4, 32)