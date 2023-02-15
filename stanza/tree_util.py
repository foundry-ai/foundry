from dataclasses import dataclass
from typing import Any

import jax

@dataclass
class Vector:
    tree: Any

    def __add__(self, rhs):
        res = jax.tree_util.tree_map(lambda a, b: a + b, self.tree, rhs.tree)
        return Vector(res)

    def __sub__(self, rhs):
        res = jax.tree_util.tree_map(lambda a, b: a - b, self.tree, rhs.tree)
        return Vector(res)