import jax.tree_util

from typing import Any

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

from jax._src.api_util import flatten_axes
from jax._src.api import _mapped_axis_size

def axis_size(pytree, axes_tree) -> int:
    args_flat, in_tree  = jax.tree_util.tree_flatten(pytree)
    in_axes_flat = flatten_axes("pvmap in_axes", in_tree, axes_tree, kws=False)
    axis_sizes_ = [x.shape[i] for x, i in zip(args_flat, in_axes_flat)]
    assert all(x == axis_sizes_[0] for x in axis_sizes_)
    return axis_sizes_[0]

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


def display_image(image, width=None, height=None):
    import numpy as np
    from PIL import Image # type: ignore
    from IPython import display
    import io
    import wandb

    if isinstance(image, wandb.Image):
        img = image._image
        if not img and image._path:
            return display.Image(filename=image._path,
                                 width=width, height=height)
    else:
        if image.dtype != np.uint8:
            image = np.array((255*image)).astype(np.uint8)
        else:
            image = np.array(image)
        img = Image.fromarray(image)
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return display.Image(data=imgByteArr,
            width=width, height=height)


class AttrMap:
    def __init__(self, *args, **kwargs):
        d = dict()
        for a in args:
            if isinstance(a, AttrMap):
                a = a._dict
            a = dict(a)
            d.update(a)
        k = dict(**kwargs)
        d.update(k)
        # Remove None values from the attr structure
        self._dict = { k: v for k,v in d.items() if v is not None }

    def __getitem__(self, name: str):
        return self._dict.__getitem__(name)
    
    def __contains__(self, name: str):
        return self._dict.__contains__(name)

    def __setattr__(self, name: str, value: Any):
        if name != '_dict':
            raise RuntimeError(f"Unable to set {name}, Attrs is immutable")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError("Not found")
        return self._dict.get(name)
    
    def set(self, k,v):
        return AttrMap(self, **{k:v})
    
    def items(self):
        return self._dict.items()
    
    def values(self):
        return self._dict.values()

    def keys(self):
        return self._dict.keys()
    
    def __repr__(self) -> str:
        return self._dict.__repr__()

    def __str__(self) -> str:
        return self._dict.__str__()
    
    def __hash__(self) -> int:
        return self._dict.__hash__()
