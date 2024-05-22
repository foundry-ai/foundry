from stanza import struct
from stanza.datasets import EnvDataset
from stanza.datasets import DatasetRegistry

from stanza.data import PyTreeData
from stanza.data.sequence import SequenceInfo, SequenceData

from .util import download, cache_path

import jax
import jax.numpy as jnp
import zarr

@struct.dataclass
class Step:
    state: jax.Array
    observation: jax.Array
    action: jax.Array

@struct.dataclass
class Chunk:
    pass

@struct.dataclass
class PushTDataset(EnvDataset[Step]):
    pass

def _load_pusht_data(zarr_path, test_trajs):
    with zarr.open(zarr_path) as zf:
        steps = Step(
            state=jnp.array(zf["data/state"]),
            observation=None,
            action=jnp.array(zf["data/action"]),
        )
        ends = jnp.array(zf["meta/episode_ends"])
        starts = jnp.roll(ends, 1).at[0].set(0)
        infos = SequenceInfo(
            id=jnp.arange(len(ends)),
            start_idx=starts,
            end_idx=ends,
            length=ends-starts,
            info=None
        )
        if test_trajs:
            train_steps = jax.tree_map(lambda x: x[:starts[-test_trajs]], steps)
            test_steps = jax.tree_map(lambda x: x[starts[-test_trajs]:], steps)
            train_infos = jax.tree_map(lambda x: x[:-test_trajs], infos)
            test_infos = jax.tree_map(lambda x: x[-test_trajs:], infos)
            test_infos = struct.replace(test_infos,
                start_idx=test_infos.start_idx - test_infos.start_idx[0],
                end_idx=test_infos.end_idx - test_infos.start_idx[0]
            )
            splits = {
                "train": SequenceData(PyTreeData(train_steps), PyTreeData(train_infos)),
                "test": SequenceData(PyTreeData(test_steps), PyTreeData(test_infos))
            }
        else:
            splits = {
                "trian": SequenceData(PyTreeData(steps), PyTreeData(infos))
            }
    return PushTDataset(
        splits=splits,
        normalizers={},
        transforms={}
    )

def _load_pusht(quiet=False):
    zip_path = cache_path("pusht", "pusht_data.zarr.zip")
    download(zip_path,
        job_name="PushT (Diffusion Policy Data)",
        gdrive_id="1ALI_Ua7U1EJRCAim5tvtQUJbBP5MGMyR",
        md5="48a64828d7f2e1e8902a97b57ebd0bdd",
        quiet=quiet
    )
    return _load_pusht_data(zip_path, 10)

registry = DatasetRegistry[PushTDataset]()
registry.register("pusht/chen", _load_pusht)