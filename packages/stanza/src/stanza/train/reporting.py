from stanza import dataclasses
import jax

class Reportable:
    pass

@dataclasses.dataclass
class Image(Reportable):
    data: jax.Array

    def _ipython_display_(self):
        from IPython.display import display
        from stanza.util.ipython import as_image
        display(as_image(self.data))

@dataclasses.dataclass
class Video(Reportable):
    data: jax.Array
    fps: int = 28

    def _ipython_display_(self):
        from IPython.display import display
        from stanza.util.ipython import as_video
        display(as_video(self.data, self.fps))

def _key_str(key):
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    elif isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    else:
        raise ValueError(f"Unknown key type: {key}")

def dict_flatten(*trees, prefix=None, suffix=None):
    flattened = {}
    for t in trees:
        paths_nodes = jax.tree_util.tree_flatten_with_path(
            t, is_leaf=lambda x: isinstance(x, Reportable))[0]
        flattened.update({
            '.'.join([_key_str(p) for p in path]): node
            for (path, node) in paths_nodes
        })
    if prefix is not None:
        flattened = {f"{prefix}{k}": v for k, v in flattened.items()}
    if suffix is not None:
        flattened = {f"{k}{suffix}": v for k, v in flattened.items()}
    return flattened

__all__ = ["Reportable", "Image", "Video", "dict_flatten"]