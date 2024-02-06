import functools
import wandb

from functools import partial
from stanza.util import dict_flatten

def wandb_log(hook=None, run=None, prefix=None, suffix=None):
    if hook is None:
        return partial(wandb_log, run=run)
    @functools.wraps(hook)
    def wrapped(rng_key, state, **kwargs):
        r = hook(rng_key, state, **kwargs)
        if r is not None and run is not None:
            flattened = dict_flatten(
                r,
                prefix=prefix, suffix=suffix
            )
            wandb.log(flattened, step=state.iteration)
    return wrapped

def wandb_stat_logger(run=None, 
                    prefix=None, suffix=None,
                    log_step_info=True):
    def log_fn(rng_key, state, **kwargs):
        stats = state.last_stats
        extra_info = {
            "iteration": state.iteration,
            "epoch": state.epoch,
            "epoch_iteration": state.epoch_iteration
        } if log_step_info else {}
        flattened = dict_flatten(
            stats, extra_info,
            prefix=prefix, suffix=suffix
        )
        wandb.log(flattened, step=state.iteration)
    return log_fn
