import jax

from dataclasses import dataclass
from typing import Any, Callable
from stanza.struct import freeze


def is_frozen(obj):
    return getattr(obj, "__frozen__", False)

# The base struct type