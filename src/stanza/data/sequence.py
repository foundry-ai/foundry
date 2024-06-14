from stanza.data import Data, PyTreeData
from typing import Any, Generic, TypeVar
from dataclasses import dataclass, field

import pickle
import jax.tree_util
import jax.numpy as jnp
import numpy as np

T = TypeVar('T')
I = TypeVar('I')

@dataclass
class SequenceInfo(Generic[I]):
    id: int
    info: I
    start_idx: int
    end_idx: int
    length: int

@dataclass
class Chunk(Generic[T,I]):
    sequence_id: int
    start_offset: int
    chunk: T
    info: I

@partial(jax.tree_util.register_dataclass,
         data_fields=['x', 'y'],
         meta_fields=['op'])
@dataclass
class ChunkData(Data, Generic[T,I]):
    elements: Data[T]
    infos: Data[SequenceInfo[I]]
    # contains the timepoints, infos offsets
    # offset by points_offset, infos_offset
    chunk_offsets: Data[tuple[int, int]]
    chunk_length: int = struct.field(pytree_node=False)

    def __len__(self) -> int:
        return len(self.chunk_offsets)
    
    def __getitem__(self, i) -> Chunk[T, I]:
        t_off, i_off = self.chunk_offsets[i]
        info = self.infos[i_off]
        chunk = self.elements.slice(t_off, self.chunk_length)
        return Chunk(
            sequence_id=info.id,
            start_offset=t_off - info.start_idx,
            chunk=chunk,
            info=info.info
        )

@struct.dataclass
class SequenceData(Data, Generic[T,I]):
    elements: Data[T]
    # contains the start, end, length
    # of each trjaectory
    # in the trajectories data
    infos: Data[SequenceInfo[I]]

    def __len__(self) -> int:
        return len(self.elements)
    
    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]
    
    def chunk(self, chunk_length: int, chunk_stride: int = 1) -> ChunkData[T,I]:
        total_chunks = 0
        infos = self.infos.slice(0, len(self.infos))
        chunks = (infos.length - chunk_length + chunk_stride) // chunk_stride
        chunks = jnp.maximum(0, chunks)
        start_chunks = jnp.cumsum(chunks) - chunks
        total_chunks = jnp.sum(chunks)
        t_off, i_off = np.zeros((2, total_chunks), dtype=jnp.int32)

        for i in range(len(self.infos)):
            idx = start_chunks[i]
            n_chunks = chunks[i]
            t_off[idx:idx+n_chunks] = infos.start_idx[i] + np.arange(n_chunks) * chunk_stride
            i_off[idx:idx+n_chunks] = i
            idx += n_chunks
        t_off, i_off = jnp.array(t_off), jnp.array(i_off)

        return ChunkData(
            elements=self.elements,
            infos=self.infos,
            chunk_offsets=PyTreeData((t_off, i_off)),
            chunk_length=chunk_length
        )

    def save(self, path):
        elements = self.elements.as_pytree()
        infos = self.infos.as_pytree()
        with open(path, "wb") as f:
            pickle.dump((elements, infos), f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            elements, infos = pickle.load(f)
            return SequenceData(
                elements=PyTreeData(elements),
                infos=PyTreeData(infos)
            )