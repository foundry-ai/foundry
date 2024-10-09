import h5py
import json
import foundry.numpy as jnp
import jax
import foundry.util

from functools import partial

from foundry.data import PyTreeData
from foundry.data.sequence import (
    SequenceInfo, SequenceData, Step
)
from foundry.env.mujoco import SystemState
from foundry.core.dataclasses import dataclass, replace

import foundry.util.serialize

from . import EnvDataset
from foundry.datasets.core import DatasetRegistry
from foundry.datasets.util import download, cache_path

@dataclass
class RobomimicDataset(EnvDataset[Step]):
    task: str = None
    dataset_type: str = None
    env_name: str = None

    def create_env(self):
        from foundry.env.mujoco.robosuite import environments
        from foundry.env.mujoco.robosuite import (
            PositionalControlTransform,
            PositionalObsTransform, RelPosObsTransform,
            RelKeypointObsTransform
        )
        from foundry.env.transforms import ChainedTransform, MultiStepTransform
        env = environments.create(self.env_name)
        env = ChainedTransform([
            PositionalControlTransform(),
            MultiStepTransform(20),
            #PositionalObsTransform()
            RelPosObsTransform()
            #RelKeypointObsTransform()
        ]).apply(env)
        return env

MD5_MAP = {
    ("can", "ph"): "758590f0916079d36fb881bd1ac5196d",
    ("square", "ph"): "ded04e6775389ca11cf77ff250b6d612",
}

def make_url(name, dataset_type):
    if dataset_type == "ph": # proficient human
        return f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{name}/ph/low_dim_v141.hdf5"
    elif dataset_type == "mh": # multi human
        return f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{name}/mh/low_dim_v141.hdf5"
    elif dataset_type == "mg": # machine generated
        return f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{name}/mg/low_dim_dense_v141.hdf5"
    elif dataset_type == "paired": # paired
        return f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{name}/paired/low_dim_v141.hdf5"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def load_robomimic_dataset(name, dataset_type, max_trajectories=None, quiet=False, full_state=True):
    """
    Load a RoboMimic dataset for a given task and dataset type.

    Args:
        name (str): name of task
        dataset_type (str): type of dataset
        max_trajectories (int): maximum number of trajectories to load
        quiet (bool): whether to suppress download progress
    
    Returns:
        env_name (str): name of environment
        data (SequenceData): sequence data containing states and actions for all trajectories
    """

    reduced_zarr_name = f"robomimic_{name}_{dataset_type}_reduced.zarr.zip"
    full_zarr_name = f"robomimic_{name}_{dataset_type}_full.zarr.zip"
    reduced_zarr_path = cache_path("robomimic", reduced_zarr_name)
    full_zarr_path = cache_path("robomimic", full_zarr_name)
    if not reduced_zarr_path.exists():
        job_name = f"robomimic_{name}_{dataset_type}.hdf5"
        hdf5_path = cache_path("robomimic", job_name)
        url = make_url(name, dataset_type)
        md5 = MD5_MAP.get((name, dataset_type), None)
        download(hdf5_path,
            job_name=job_name,
            url=url, md5=md5,
            quiet=quiet
        )
        env_name, data = _load_robomimic_hdf5(hdf5_path, max_trajectories)
        foundry.util.serialize.save_zarr(reduced_zarr_path, 
            tree=data, meta=env_name
        )
        hdf5_path.unlink()
    if full_state and not full_zarr_path.exists():
        data, env_name = foundry.util.serialize.load_zarr(reduced_zarr_path)
        from foundry.env.mujoco.robosuite import environments
        env = environments.create(env_name)
        data = data.map_elements(lambda x: replace(x, 
            state=env.full_state(x.reduced_state)
        )).cache()
        foundry.util.serialize.save_zarr(full_zarr_path, 
            tree=data, meta=env_name
        )
    data, env_name = foundry.util.serialize.load_zarr(full_zarr_path if full_state else reduced_zarr_path)
    return env_name, data

ENV_MAP = {
    "PickPlaceCan": "pickplace/can",
    "PickPlaceMilk": "pickplace/milk",
    "PickPlaceBread": "pickplace/bread",
    "PickPlaceCereal": "pickplace/cereal",
    "NutAssemblySquare": "nutassembly/square",
    "NutAssemblyRound": "nutassembly/round"
}

def _load_robomimic_hdf5(hdf5_path, max_trajectories=None):
    """
    Load a RoboMimic dataset from an HDF5 file.
    Returns:
        env_meta (dict): environment metadata, which should be loaded from demonstration
                hdf5. Contains 3 keys:

                    :`'env_name'`: name of environment
                    :`'type'`: type of environment, should be a value in EB.EnvType
                    :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
        SequenceData containing states and actions for all trajectories
    """
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_meta = json.loads(data.attrs["env_args"])
        # Create the environment in order
        # to determine the dimensions of the qpos, qvel components
        from foundry.env.mujoco.robosuite import environments
        env = environments.create(ENV_MAP[env_meta["env_name"]])
        reduced_state = env.reduce_state(env.sample_state(jax.random.key(42)))
        nq, = reduced_state.qpos.shape
        nqv, = reduced_state.qvel.shape

        states = None
        actions = None
        start_idx = []
        end_idx = []
        length = []
        for traj in data: 
            traj_states = jnp.asarray(data[traj]["states"][:,:])
            T = traj_states.shape[0]
            traj_states = SystemState(
                traj_states[:,0], traj_states[:,1:1+nq],
                traj_states[:,1+nq:1+nq+nqv],
                jnp.zeros((T,0), dtype=jnp.float32)
            )
            traj_actions = jnp.asarray(data[traj]["actions"][:,:])
            if states is None: states = traj_states
            else: states = jax.tree.map(lambda a, b: jnp.concatenate((a,b)), states, traj_states)
            if actions is None: actions = traj_actions
            else: actions = jnp.concatenate((actions, traj_actions))
            start_idx.append(actions.shape[0] - T)
            end_idx.append(actions.shape[0])
            length.append(T)

        start_idx = jnp.array(start_idx) # remove last start index
        end_idx = jnp.array(end_idx)
        length = jnp.array(length)
        infos = SequenceInfo(
            start_idx=start_idx,
            end_idx=start_idx+length,
            length=length,
            info=None
        )
        steps = Step(
            state=None,
            reduced_state=states,
            observation=None,
            action=actions
        )
        return ENV_MAP[env_meta["env_name"]], SequenceData(PyTreeData(steps), PyTreeData(infos))

def load_robomimic(*, name=None, dataset_type=None, quiet=False, **kwargs):
    if name is None or dataset_type is None:
        raise ValueError("Must specify a task, dataset_type to load robomimic dataset.")
    env_name, data = load_robomimic_dataset(
        name=name, dataset_type=dataset_type, quiet=quiet
    )
    train = data.slice(0, len(data) - 32)
    test = data.slice(len(data) - 16, 16)
    validation = data.slice(len(data) - 32, 16)
    return RobomimicDataset(
        splits={
            "train": train, "test": test, 
            "validation": validation
        },
        env_name=env_name
    )

def register_all(registry: DatasetRegistry, prefix=None):
    for dataset_type in ["ph", "mh", "mg"]:
        for task in ["pickplace", "nutassembly"]:
            if task == "pickplace":
                for obj in ["can", "milk", "bread", "cereal"]:
                    name = obj
                    registry.register(f"robomimic/{task}/{obj}/{dataset_type}", 
                        partial(load_robomimic,name=name, dataset_type=dataset_type)
                    )
            elif task == "nutassembly":
                for obj in ["square", "round"]:
                    name = obj
                    registry.register(f"robomimic/{task}/{obj}/{dataset_type}", 
                        partial(load_robomimic,name=name, dataset_type=dataset_type)
                    )