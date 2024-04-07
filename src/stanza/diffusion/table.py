from stanza import struct
from stanza.util.pca import randomized_pca
from stanza.util.random import PRNGSequence

from typing import Generic, TypeVar
from typing import Tuple, NamedTuple

import stanza.util
import jax
import jax.numpy as jnp

import numpy as np

L = TypeVar('L')


@struct.dataclass
class TableNode:
    # 00s if leaf
    children: jax.Array
    n: jax.Array
    dir: jax.Array
    dir_sigma: jax.Array
    split_value: jax.Array
    mean: jax.Array

@struct.dataclass
class DiffusionTable:
    nodes: TableNode

    # Does a nearest-neighbor search
    # of the n_components that are within sigma
    def query_components(self, leaf: L, sigma: jax.Array,
                        n_components: int):
        pass

    @staticmethod
    def build(leaves: L) -> "DiffusionTable[L]":
        rng = PRNGSequence(jax.random.PRNGKey(42))
        flat_leaves = jax.vmap(lambda x: stanza.util.ravel_pytree(x)[0])(leaves)
        temp_nodes = []
        queue = []
        queue.append((-1, 0, 0, len(flat_leaves)))
        while queue:
            parent_idx, child_idx, start, end = queue.pop()
            # print("processing", parent_idx, child_idx, start, end)
            leaves = flat_leaves[start:end] # re-order the leaves
            mean = jnp.mean(leaves, axis=0)
            # rearrange the leaves for this node
            if leaves.shape[0] > 1:
                leaves, partition_idx, dir, dir_sigma, split_value = \
                    find_split(leaves, next(rng))
                partition_idx = int(partition_idx)
                flat_leaves = flat_leaves.at[start:end].set(leaves)

                # will re-arrange the leaves for this node
                # according to the split direction
                new_parent_idx = len(temp_nodes)
                temp_nodes.append(NodeTemp(
                    parent=parent_idx, child_idx=child_idx, start=start, end=end,
                    dir=dir, split_value=split_value, mean=mean
                ))
                if partition_idx > 0 and partition_idx < leaves.shape[0]:
                    queue.append((new_parent_idx, 0,
                                start, start + partition_idx))
                    queue.append((new_parent_idx, 1,
                                start + partition_idx, end))
            else:
                temp_nodes.append(NodeTemp(
                    parent=parent_idx, child_idx=child_idx, start=start, end=end,
                    dir=jnp.zeros_like(mean),
                    dir_sigma=jnp.zeros(()),
                    split_value=jnp.zeros(()),
                    mean=mean
                ))
        # the (left, right) children indices
        n_nodes = len(temp_nodes)
        nodes = TableNode(
            children=np.zeros((n_nodes, 2), dtype=np.uint64),
            dir=np.zeros((n_nodes, flat_leaves.shape[-1])),
            split_value=np.zeros((n_nodes,)),
            mean=np.zeros((n_nodes, flat_leaves.shape[-1])),
        )
        for i, n in enumerate(temp_nodes):
            # set the nth node's children
            if n.parent >= 0:
                nodes.children[n.parent, n.child_idx] = i
            nodes.dir[i] = n.dir
            nodes.n[i] = n.end - n.start
            nodes.dir_sigma[i] = n.dir_sigma
            nodes.split_value[i] = n.split_value
            nodes.mean[i] = n.mean
        nodes = jax.tree_map(lambda x: jnp.asarray(x), nodes)
        return DiffusionTable(nodes)

class NodeTemp(NamedTuple):
    parent: int
    child_idx: int
    start: int
    end: int
    dir: jax.Array
    dir_sigma: jax.Array
    split_value: float
    mean: jax.Array

@jax.jit
def find_split(leaves, rng):
    # randomly pick
    num_svd = min(leaves.shape[0], 64)
    l_rng, p_rng = jax.random.split(rng)
    random_leaves = jax.random.choice(l_rng, leaves, shape=(num_svd,), replace=True)
    pca = randomized_pca(random_leaves, 1, p_rng)
    v = pca.components[0]
    leaf_vs = jax.vmap(lambda x: jnp.dot(v, x))(leaves)
    partition_idx = leaf_vs.shape[0] // 2
    partition_indices = jnp.argpartition(leaf_vs, leaf_vs.shape[0] // 2)
    partition_leaves = jnp.take(leaves, partition_indices, axis=0)
    split_value = leaf_vs[partition_indices[partition_idx]]
    value_std = jnp.std(leaf_vs)
    return partition_leaves, partition_idx, v, value_std, split_value