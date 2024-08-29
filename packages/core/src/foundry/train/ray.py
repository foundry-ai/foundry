from foundry.train.reporting import as_log_dict

from functools import partial
from ray import train

import jax

def _log_cb(iteration, data_dict):
    data = {k: v.item() for k, v in data_dict.items()}
    train.report(data)

@partial(jax.jit,
        static_argnames=("join", "prefix", "suffix")
    )
def report(iteration, *data, join=".", prefix=None, suffix=None):
    data, _ = as_log_dict(*data, join=join, prefix=prefix, suffix=suffix)
    from ray.train._internal.session import _get_session
    if _get_session() is None:
        return
    jax.experimental.io_callback(
        _log_cb, None, 
        iteration, data, ordered=True
    )