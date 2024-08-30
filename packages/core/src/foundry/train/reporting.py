from foundry.core.dataclasses import dataclass

from foundry.core import tree
import jax

class Reportable:
    pass

@dataclass
class Image(Reportable):
    data: jax.Array

    def _ipython_display_(self):
        from IPython.display import display
        from foundry.util.ipython import as_image
        display(as_image(self.data))

@dataclass
class Video(Reportable):
    data: jax.Array
    fps: int = 28

    def _ipython_display_(self):
        from IPython.display import display
        from foundry.util.ipython import as_video
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

def as_log_dict(*trees, join=".", prefix=None, suffix=None):
    data = {}
    reportables = {}
    for t in trees:
        data_entries = jax.tree.map(
            lambda x: (None if isinstance(x, Reportable) else x), 
            t, is_leaf=lambda x: isinstance(x, Reportable)
        )
        reportable_entries = jax.tree.map(
            lambda x: x if isinstance(x, Reportable) else None, 
            t, is_leaf=lambda x: isinstance(x, Reportable)
        )
        # flatten data_entries, reportable_entries
        # to dictionaries
        data_entries, _ = tree.flatten_to_dict(data_entries,
            join=join, prefix=prefix, suffix=suffix,
            is_leaf=lambda x: isinstance(x, Reportable)
        )
        reportable_entries, _ = tree.flatten_to_dict(reportable_entries,
            join=join, prefix=prefix, suffix=suffix,
            is_leaf=lambda x: isinstance(x, Reportable)
        )
        data.update(data_entries)
        reportables.update(reportable_entries)
    return data, reportables

__all__ = ["Reportable", "Image", "Video", "as_log_dict"]