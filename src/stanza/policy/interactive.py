from stanza import dataclasses

import jax
import jax.numpy as jnp

KEY_MAP = {
    chr(ord('a') + i): i for i in range(26)
}