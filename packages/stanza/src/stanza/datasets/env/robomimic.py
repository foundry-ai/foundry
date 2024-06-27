import h5py
import json
from stanza.data import PyTreeData
from stanza.data.sequence import (
    SequenceInfo, SequenceData, Step
)
from stanza.dataclasses import dataclass
from stanza.datasets import EnvDataset, DatasetRegistry
import jax.numpy as jnp

from ..util import download, cache_path


# @dataclass
# class RobomimicDataset(EnvDataset[Step]):
#     task: str
#     dataset_type: str
#     env_meta: dict

#     def create_env(self):
#         from stanza.env.mujoco.robomimic import RobomimicEnv
#         env = RobomimicEnv(self.task, self.dataset_type, self.env_meta)
#         return env
    



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
    job_name = f"robomimic_{task}_{dataset_type}"
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

        dim_state = jnp.asarray(data["demo_0"]["states"][:,:]).shape[-1]
        states = jnp.array([]).reshape(0,dim_state)
        dim_action = jnp.asarray(data["demo_0"]["actions"][:,:]).shape[-1]
        actions = jnp.array([]).reshape(0,dim_action)


        start_idx = [0]
        length = []
        for traj in data: 
            traj_states = jnp.asarray(data[traj]["states"][:,:])
            traj_actions = jnp.asarray(data[traj]["actions"][:,:])
            states = jnp.concatenate((states, traj_states))
            actions = jnp.concatenate((actions, traj_actions))
            start_idx.append(states.shape[0])
            length.append(states.shape[0])
        start_idx = jnp.array(start_idx[:-1]) # remove last start index
        length = jnp.array(length)
        infos = SequenceInfo(
            start_idx=start_idx,
            end_idx=start_idx+length,
            length=length,
            info=None
        )
        steps = Step(
            state=states,
            observation=None,
            action=actions
        )
        return env_meta, SequenceData(PyTreeData(steps), PyTreeData(infos))

# datasets = DatasetRegistry[RobomimicDataset]()
# datasets.register("robomimic", load_robomimic)

