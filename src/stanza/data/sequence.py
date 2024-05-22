from stanza.data import Data, PyTreeData
from stanza import struct

from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar('T')
I = TypeVar('I')

@struct.dataclass
class SequenceInfo(Generic[I]):
    id: int
    info: I
    start_idx: int
    end_idx: int
    length: int

@struct.dataclass
class Chunk(Generic[T,I]):
    sequence_id: int
    start_offset: int
    chunk: T
    info: I

@struct.dataclass
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
        for i in range(len(self.infos)):
            info = self.infos[i]
            chunks = (info.length - chunk_length + chunk_stride) // chunk_stride
            total_chunks += max(0, chunks)
        t_off, i_off = np.zeros((2, total_chunks), dtype=jnp.int32)
        idx = 0
        for i in range(len(self.infos)):
            info = self.infos[i]
            chunks = (info.length - chunk_length + chunk_stride) // chunk_stride
            t_off[idx:idx+chunks] = info.start_idx + np.arange(chunks) * chunk_stride
            i_off[idx:idx+chunks] = i
            idx += chunks

        return ChunkData(
            elements=self.elements,
            infos=self.infos,
            chunk_offsets=PyTreeData((t_off, i_off)),
            chunk_length=chunk_length
        )