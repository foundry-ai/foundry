import functools
import wandb

from functools import partial
from stanza.util import dict_flatten
from pathlib import Path

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

def wandb_checkpoint(run=None, dir="checkpoints",
                format="epoch_{epoch}.ckpt"):
    import orbax.checkpoint as ocp

    run = run or wandb.run
    dir = Path(run.dir) / dir
    checkpointer = ocp.StandardCheckpointer()

    def log_fn(rng_key, state, **kwargs):
        vars = state.vars
        stats = state.last_stats
        extra_info = {
            "iteration": state.iteration,
            "epoch": state.epoch,
        }
        metadata = dict_flatten(stats, extra_info)
        name = format.format(**extra_info)
        path = dir / name
        checkpointer.save(path, vars)
        # log to wandb
        artifact = wandb.Artifact(name, "model", metadata=metadata)
        if path.is_dir():
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        run.log_artifact(artifact)
    return log_fn