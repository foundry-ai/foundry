from stanza import struct
import jax

class Reportable:
    pass

@struct.dataclass
class Image(Reportable):
    data: jax.Array

    def __ipython_display__(self):
        from IPython.display import display
        from stanza.util.ipython import as_image
        display(as_image(self.data))

@struct.dataclass
class Video(Reportable):
    data: jax.Array
    fps: int = 28

    def __repr_mimebundle__(self, include=None, exclude=None):
        from IPython.display import display
        from stanza.util.ipython import as_video
        display(as_video(self.data, self.fps))

def _key_str(key):
    return key.key

def dict_flatten(*trees, prefix=None, suffix=None):
    flattened = {}
    for t in trees:
        paths_nodes = jax.tree_util.tree_flatten_with_path(
            t, is_leaf=lambda x: not isinstance(x, Reportable))[0]
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