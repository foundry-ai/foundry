from typing import Any
from stanza.util.dataclasses import dataclass

import jax

# Will rescale to [-1, 1]
@dataclass(jax=True)
class LinearRescale:
    min: Any
    max: Any

    def scale_data(data):
        return jax.tree_util.tree_map()

    @staticmethod
    def from_data(data):
        data.data