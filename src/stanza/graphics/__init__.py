import math
import jax.numpy as jnp
import jax

from functools import partial

def sanitize_color(color, channels=4):
    color = jnp.atleast_1d(jnp.array(color))
    if color.shape[-1] == 1 and channels >= 3:
        color = color.repeat(3, axis=-1)
    if color.shape[-1] == channels:
        return color
    # if alpha is not specified, make alpha = 1
    elif color.shape[-1] == 3 and channels == 4 or \
            color.shape[-1] == 1 and channels == 2:
        return jnp.concatenate((color, jnp.ones(color.shape[:-1] + (1,))), axis=-1)
    elif color.shape[-1] == 4 and channels == 3:
        return color[:3]
    elif channels == 1:
        return jnp.atleast_1d(jnp.mean(color))
    else:
        raise ValueError("Invalid color shape")

@partial(jax.jit, static_argnums=(1,))
def pad(img, padding=1, color=0):
    color = sanitize_color(color, channels=img.shape[-1])
    def do_pad(channel, value):
        return jnp.pad(channel, 
            padding, constant_values=value)
    return jax.vmap(do_pad, in_axes=-1, out_axes=-1)(img, color)

@partial(jax.jit, static_argnums=(1,2))
def image_grid(images, cols=None, rows=None):
    N = images.shape[0]
    if N == 1:
        return images[0]
    has_channels = len(images.shape) == 4

    # use a heuristic to pick a good
    # number of rows and columns
    if cols is None and rows is None:
        diff = math.inf
        for c in range(1,min(N+1, 10)):
            r = math.ceil(N / c)
            n_diff = abs(c-r) + 5*abs(N - r*c)
            if n_diff <= diff:
                rows = r
                cols = c
                diff = n_diff
    if cols is None:
        cols = math.ceil(N / rows)
    if rows is None:
        rows = math.ceil(N / cols)

    # add zero padding for missing images
    if rows*cols > N:
        padding = jnp.zeros((rows*cols - N,) + images.shape[1:],
                            dtype=images.dtype)
        images = jnp.concatenate((images, padding), axis=0)
    images = jnp.reshape(images, (rows, cols,) + images.shape[1:])
    # reorder row, cols, height, width, channels 
    # to row, height, cols, width, channels
    images = jnp.transpose(images,
        (0, 2, 1, 3, 4)
        if has_channels else
        (0, 2, 1, 3)
    )
    # reshape to flatten the columns
    images = jnp.reshape(images, 
        (images.shape[0], images.shape[1], -1, images.shape[4])
        if has_channels else
        (images.shape[0], images.shape[1], -1)
    )
    # reshape to flatten the rows
    images = jnp.reshape(images,
        (-1, images.shape[2], images.shape[3])
        if has_channels else
        (-1, images.shape[2])
    )
    return images
