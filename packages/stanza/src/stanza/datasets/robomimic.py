import h5py

from .util import download, cache_path

# all proficient human datasets
ph_tasks = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]

# all multi human datasets
mh_tasks = ["lift", "can", "square", "transport"]


# all machine generated datasets
mg_tasks = ["lift", "can"]

# can-paired dataset
"http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/low_dim_v141.hdf5"

def load_robomimic_dataset(task, dataset_type, max_trajectories=None, quiet=False):
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
    return load_robomimic_hdf5(hdf5_path, max_trajectories)

def load_robomimic_hdf5(hdf5_path, max_trajectories=None):
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        for traj in data: 

