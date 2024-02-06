import math
import jax.tree_util
import jax.numpy as jnp

MISSING = object()

class FrozenInstanceError(AttributeError): pass

def _key_str(key):
    return key.key

def dict_flatten(*trees, prefix=None, suffix=None):
    flattened = {}
    for t in trees:
        paths_nodes = jax.tree_util.tree_flatten_with_path(t)[0]
        flattened.update({
            '.'.join([_key_str(p) for p in path]): node
            for (path, node) in paths_nodes
        })
    if prefix is not None:
        flattened = {f"{prefix}{k}": v for k, v in flattened.items()}
    if suffix is not None:
        flattened = {f"{k}{suffix}": v for k, v in flattened.items()}
    return flattened


def grid(images, cols=None, rows=None):
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

from rich.text import Text as RichText
from rich.progress import ProgressColumn

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return RichText(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )