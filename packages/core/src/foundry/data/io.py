import foundry.numpy as jnp
import jax
import numpy as np

from PIL import Image

def read_image(path, expected_shape, /, *args, **kwargs):
    path = str(path)
    def read(args, kwargs):
        f = path.format(*args, **kwargs)
        data = jnp.array(Image.open(f), dtype=jnp.uint8)
        assert data.shape == expected_shape, data.dtype == jnp.uint8
        return data
    return jax.pure_callback(read,
        jax.ShapeDtypeStruct(expected_shape, jnp.uint8),
        args, kwargs
    )