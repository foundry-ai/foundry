import jax.tree_util

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