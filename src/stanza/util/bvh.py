from stanza import struct
from stanza.util.pca import randomized_pca
from stanza.util.random import PRNGSequence

from typing import Generic, TypeVar
from typing import Tuple

import stanza.util
import jax
import jax.numpy as jnp

L = TypeVar('L')

@struct.dataclass
class DiffusionTable:
    @staticmethod
    def build(leaves: L) -> "DiffusionTable[L]":
        rng = PRNGSequence(jax.random.PRNGKey(42))
        flat_leaves = jax.vmap(lambda x: stanza.util.ravel_pytree(x)[0])(leaves)
        nodes = []
        queue = []
        queue.append((0, len(flat_leaves)))

        def process_node(x, rng):
            # randomly pick
            leaves = x[start:end]
            num_svd = min(leaves.shape[0], 64)
            l_rng, p_rng = jax.random.split(rng)
            random_leaves = jax.random.choice(l_rng, leaves, shape=(num_svd,), replace=True)
            pca = randomized_pca(random_leaves, 1, p_rng)
            v = pca.components[0]
            leave_vs = jax.vmap(lambda x: jnp.dot(v, x))(leaves)
            partition_indices = jnp.argpartition(leave_vs, leave_vs.shape[0] // 2)
            partition_leaves = jnp.take_along_axis(leaves, partition_indices, axis=0)
            return partition_leaves

            # get the vs and split by two
        while queue:
            start, end = queue.pop()
            if end - start > 1:
                pass
            leaves = flat_leaves[start:end]
            leaves = process_node(leaves, next(rng))
            # will re-arrange the leaves for this node
            # according to the split direction
            flat_leaves = flat_leaves.at[start:end].set(leaves)
        return DiffusionTable(leaves)