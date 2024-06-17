from stanza import dataclasses
from stanza.datasets import EnvDataset
from stanza.datasets import DatasetRegistry

from stanza.data import PyTreeData
from stanza.data.sequence import SequenceInfo, SequenceData

from .util import download, cache_path

import jax
import jax.numpy as jnp
import zarr

@dataclasses.dataclass
class Step:
    state: jax.Array
    observation: jax.Array
    action: jax.Array

@dataclasses.dataclass
class Chunk:
    pass

@dataclasses.dataclass
class PushTDataset(EnvDataset[Step]):
    def create_env(self):
        from stanza.env.mujoco.pusht import PushTEnv
        return PushTEnv()

def _load_pusht_data(zarr_path, train_trajs, test_trajs):
    with zarr.open(zarr_path) as zf:
        if train_trajs is None:
            train_trajs = len(zf["meta/episode_ends"]) - test_trajs
        total_trajs = train_trajs + test_trajs

        ends = jnp.array(zf["meta/episode_ends"])
        starts = jnp.roll(ends, 1).at[0].set(0)

        infos = SequenceInfo(
            id=jnp.arange(len(ends)),
            start_idx=starts,
            end_idx=ends,
            length=ends-starts,
            info=None
        )
        steps = Step(
            state=zf["data/state"],
            observation=None,
            action=zf["data/action"]
        )
        # slice the data into train and test
        train_steps = jax.tree_map(lambda x: jnp.array(x[:starts[-test_trajs]]), steps)
        test_steps = jax.tree_map(lambda x: jnp.array(x[starts[-test_trajs]:]), steps)
        train_infos = jax.tree_map(lambda x: x[:-test_trajs], infos)
        test_infos = jax.tree_map(lambda x: x[-test_trajs:], infos)
        # Adjust the start and end indices to be relative to the first episode
        test_infos = dataclasses.replace(test_infos,
            start_idx=test_infos.start_idx - test_infos.start_idx[0],
            end_idx=test_infos.end_idx - test_infos.start_idx[0]
        )
        train_infos = dataclasses.replace(train_infos,
            start_idx=train_infos.start_idx - train_infos.start_idx[0],
            end_idx=train_infos.end_idx - train_infos.start_idx[0]
        )
        splits = {
            "train": SequenceData(PyTreeData(train_steps), PyTreeData(train_infos)),
            "test": SequenceData(PyTreeData(test_steps), PyTreeData(test_infos))
        }
    return PushTDataset(
        splits=splits,
        normalizers={},
        transforms={}
    )

def _load_pusht(quiet=False, train_trajs=None, test_trajs=10):
    zip_path = cache_path("pusht", "pusht_data.zarr.zip")
    download(zip_path,
        job_name="PushT (Diffusion Policy Data)",
        gdrive_id="1ALI_Ua7U1EJRCAim5tvtQUJbBP5MGMyR",
        md5="48a64828d7f2e1e8902a97b57ebd0bdd",
        quiet=quiet
    )
    return _load_pusht_data(zip_path, train_trajs, test_trajs)

registry = DatasetRegistry[PushTDataset]()
registry.register("pusht/chen", _load_pusht)