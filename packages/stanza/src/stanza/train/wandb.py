import jax
import functools
import wandb
import numpy as np

from functools import partial
from stanza.train.reporting import Video, Image, as_log_dict
from pathlib import Path

def _map_reportable(x):
    if isinstance(x, Image):
        return wandb.Image(np.array(x.data))
    elif isinstance(x, Video):
        return wandb.Video(x.data)
    return None

def _log_cb(run, iteration, data_dict, reportable_dict):
    iteration = int(iteration)
    run = run if run is not None else wandb.run
    items = dict({k: v.item() for (k, v) in data_dict.items()})
    for k, v in reportable_dict.items():
        v = _map_reportable(v)
        items[k] = v
    run.log(items, step=iteration)

@partial(jax.jit,
        static_argnames=("run", "join", "prefix", "suffix")
    )
def log(iteration, *data, run=None, join=".", prefix=None, suffix=None):
    data, reportables = as_log_dict(*data, join=join, prefix=prefix, suffix=suffix)
    jax.experimental.io_callback(
        partial(_log_cb, run), None, 
        iteration, data, reportables,
        ordered=True
    )