import jax
import numpy as np
import tempfile
import warnings

from stanza.train.reporting import Video, Image, as_log_dict
from functools import partial

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ffmpegio

def map_reportable(x):
    import wandb
    if isinstance(x, Image):
        return wandb.Image(np.array(x.data))
    elif isinstance(x, Video):
        array = np.array(x.data)
        array = np.nan_to_num(array, copy=False, 
                            nan=0, posinf=0, neginf=0)
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array*255).clip(0, 255).astype(np.uint8)
        f = tempfile.mktemp() + ".mp4"
        ffmpegio.video.write(f, x.fps, array)
        return wandb.Video(f)
    return None

def _log_cb(run, iteration, data_dict, reportable_dict):
    import wandb
    iteration = int(iteration)
    run = run if run is not None else wandb.run
    items = dict({k: v.item() for (k, v) in data_dict.items()})
    for k, v in reportable_dict.items():
        v = map_reportable(v)
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