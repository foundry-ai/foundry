import jax.numpy as jnp
import jax

from PIL import Image

def read_image(path, expected_shape, /, *args, **kwargs):
    path = str(path)
    def read(args, kwargs):
        f = path.format(*args, **kwargs)
        return jnp.array(Image.open(f), dtype=jnp.uint8)
    return jax.pure_callback(read,
        jax.ShapeDtypeStruct(expected_shape, jnp.uint8),
        args, kwargs
    )