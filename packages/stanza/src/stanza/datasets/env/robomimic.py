import h5py
import json
import jax.numpy as jnp
import jax
import stanza.util

from functools import partial

from stanza.data import PyTreeData
from stanza.data.sequence import (
    SequenceInfo, SequenceData, Step
)
from stanza.env.mujoco import SystemState
from stanza.dataclasses import dataclass

from . import EnvDataset
from .. import DatasetRegistry
from ..util import download, cache_path

@dataclass
class RobomimicDataset(EnvDataset[Step]):
    task: str = None
    dataset_type: str = None
    env_name: str = None

    def create_env(self):
        from stanza.env.mujoco.robosuite import environments
        return environments.create(self.env_name)

def load_robomimic_dataset(task, dataset_type, max_trajectories=None, quiet=False):
    """
    Load a RoboMimic dataset for a given task and dataset type.

    all proficient human datasets:
    ph_tasks = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]

    all multi human datasets:
    mh_tasks = ["lift", "can", "square", "transport"]

    all machine generated datasets:
    mg_tasks = ["lift", "can"]

    Returns:
        env_meta (dict): environment metadata, which should be loaded from demonstration
                hdf5. Contains 3 keys:

                    :`'env_name'`: name of environment
                    :`'type'`: type of environment, should be a value in EB.EnvType
                    :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
        SequenceData: contains states and actions for all trajectories
    """
    job_name = f"robomimic_{task}_{dataset_type}.hdf5"
    hdf5_path = cache_path("robomimic", job_name)
    if dataset_type == "ph": # proficient human
        url = f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{task}/ph/low_dim_v141.hdf5"
    elif dataset_type == "mh": # multi human
        url = f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{task}/mh/low_dim_v141.hdf5"
    elif dataset_type == "mg": # machine generated
        url = f"http://downloads.cs.stanford.edu/downloads/rt_benchmark/{task}/mg/low_dim_dense_v141.hdf5"
    elif dataset_type == "paired": # paired
        url = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/low_dim_v141.hdf5"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    download(hdf5_path,
        job_name=job_name,
        url=url,
        quiet=quiet
    )
    return _load_robomimic_hdf5(hdf5_path, max_trajectories)

ENV_MAP = {
    "PickPlaceCan": "can"
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
        from stanza.env.mujoco.robosuite import environments
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
            reduced_state=states,
            observation=None,
            action=actions
        )
        return env_meta, SequenceData(PyTreeData(steps), PyTreeData(infos))
    
def load_robomimic(*, task=None, dataset_type=None, quiet=False, **kwargs):
    if task is None or dataset_type is None:
        raise ValueError("Must specify a task, dataset_type to load robomimic dataset.")
    env_meta, data = load_robomimic_dataset(
        task=task, dataset_type=dataset_type, quiet=quiet
    )
    
    return RobomimicDataset(
        splits={"train": data},
        env_name=ENV_MAP[env_meta["env_name"]],
    )

datasets = DatasetRegistry[RobomimicDataset]()
datasets.register("can/ph", partial(load_robomimic,task="can",dataset_type="ph"))