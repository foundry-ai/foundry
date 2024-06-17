import dataclasses as dcls
import jax

from dataclasses import MISSING, replace
from functools import partial

def fields(cls):
    return dcls.fields(cls)

def dataclass(cls=None, **kwargs):
    if cls is None:
        return partial(dataclass, **kwargs)
    cls = dcls.dataclass(frozen=True, **kwargs)(cls)
    fields = dcls.fields(cls)
    data_fields = [f.name for f in fields if (not f.metadata) and f.metadata.get('pytree_node', True)]
    meta_fields = [f.name for f in fields if f.metadata and (not f.metadata.get('pytree_node', True))]
    cls = jax.tree_util.register_dataclass(cls, data_fields, meta_fields)
    return cls

def field(*, default=MISSING, default_factory=MISSING, pytree_node=True):
    return dcls.field(
        default=default, default_factory=default_factory,
        metadata={'pytree_node': pytree_node}
    )