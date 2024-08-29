from foundry.core.dataclasses import dataclass, replace
from foundry.datasets import DatasetRegistry
from foundry.datasets.env import EnvDataset

from foundry.data import PyTreeData, idx_dtype
from foundry.data.sequence import (
    SequenceInfo, SequenceData, Step
)

from ..util import download, cache_path

import foundry.util.serialize
import jax
import foundry.numpy as jnp
import zarr

@dataclass
class PushTDataset(EnvDataset[Step]):
    def create_env(self, **kwargs):
        from foundry.env.mujoco.pusht import (
            PushTEnv,
            PositionalControlTransform,
            KeypointObsTransform,
        )
        from foundry.env.transforms import ChainedTransform, MultiStepTransform
        env = PushTEnv()
        env = ChainedTransform([
            PositionalControlTransform(),
            MultiStepTransform(10),
            #PositionalObsTransform()
            KeypointObsTransform()
            #RelKeypointObsTransform()
        ]).apply(env)
        return env

def load_pytorch_pusht_data(zarr_path, max_trajectories=None):
    from foundry.env.mujoco import SystemState
    with zarr.open(zarr_path) as zf:
        if max_trajectories is None:
            max_trajectories = len(zf["meta/episode_ends"])
        ends = jnp.array(zf["meta/episode_ends"][:max_trajectories], dtype=idx_dtype)
        starts = jnp.roll(ends, 1).at[0].set(0)
        last_end = ends[-1]
        infos = SequenceInfo(
            start_idx=starts,
            end_idx=ends,
            length=ends-starts,
            info=None
        )

        @jax.vmap
        def convert_states(state):
            agent_pos = jnp.array([1, -1], dtype=jnp.float32)*((state[:2] - 256) / 252)
            block_pos = jnp.array([1, -1], dtype=jnp.float32)*((state[2:4] - 256) / 252)
            block_rot = -state[4]
            # our rotation q is around the block center of mass
            # while theirs is around block_pos
            # we need to adjust the position accordingly
            # our_true_block_pos = our_block_body_q_pos + com_offset - our_q_rot @ com_offset
            # we substitute our_true_pos for block_pos and solve
            rotM = jnp.array([
                [jnp.cos(block_rot), -jnp.sin(block_rot)],
                [jnp.sin(block_rot), jnp.cos(block_rot)]
            ], dtype=jnp.float32)
            block_scale = 30/252
            com = 0.5*(block_scale/2) + 0.5*(2.5*block_scale)
            com = jnp.array([0, -com], dtype=jnp.float32)
            block_pos = block_pos + rotM @ com - com

            q = jnp.concatenate([agent_pos, block_pos, block_rot[None]])
            return SystemState(
                time=jnp.ones((), dtype=jnp.float32), 
                qpos=q, qvel=jnp.zeros_like(q),
                act=jnp.zeros((0,), dtype=jnp.float32)
            )

        @jax.vmap
        def convert_actions(action):
            return action / 256 - 1

        steps = Step(
            reduced_state=convert_states(jnp.array(zf["data/state"][:last_end], dtype=jnp.float32)),
            observation=None,
            action=convert_actions(jnp.array(zf["data/action"][:last_end], dtype=jnp.float32))
        )
    return SequenceData(PyTreeData(steps), PyTreeData(infos))

def load_chi_pusht_data(max_trajectories=None, quiet=False):
    zip_path = cache_path("pusht", "pusht_data_raw.zarr.zip")
    processed_path = cache_path("pusht", "pusht_data.zarr.zip")
    if not processed_path.exists():
        download(zip_path,
            job_name="PushT (Diffusion Policy Data)",
            gdrive_id="1ALI_Ua7U1EJRCAim5tvtQUJbBP5MGMyR",
            md5="48a64828d7f2e1e8902a97b57ebd0bdd",
            quiet=quiet
        )
        data = load_pytorch_pusht_data(zip_path, max_trajectories)
        foundry.util.serialize.save_zarr(processed_path, data, None)
        # remove the raw data
        zip_path.unlink()
    data, _ = foundry.util.serialize.load_zarr(processed_path)
    return data

def load_chi_pusht(quiet=False, train_trajs=None, test_trajs=10):
    data = load_chi_pusht_data()
    train = data.slice(0, len(data) - 32)
    validation = data.slice(len(data) - 32, 16)
    test = data.slice(len(data) - 16, 16)
    return PushTDataset(
        splits={"train": train, "validation": validation, "test": test},
        normalizers={},
        transforms={}
    )

datasets = DatasetRegistry[PushTDataset]()
datasets.register(load_chi_pusht)
datasets.register("chi", load_chi_pusht)