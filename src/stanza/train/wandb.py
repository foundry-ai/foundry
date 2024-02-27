import functools
import wandb

from functools import partial
from stanza.train.reporting import Video, Image, dict_flatten
from pathlib import Path

def convert(x):
    if isinstance(x, Image):
        return wandb.Image(x.data)
    elif isinstance(x, Video):
        return wandb.Video(x.data)
    return x

def wandb_log(*hooks, run=None, prefix=None, 
              suffix=None, metrics=False, step_info=False):
    def logger(rng, state, log=None, **kwargs):
        if run is not None:
            r = []
            if log is not None: r.append(log)
            if metrics: r.append(state.metrics)
            if step_info: r.append({
                "iteration": state.iteration,
                "epoch": state.epoch,
                "epoch_iteration": state.epoch_iteration
            })
            for hook in hooks: r.append(hook(rng, state, **kwargs))
            flattened = dict_flatten(
                *r,
                prefix=prefix, suffix=suffix
            )
            wandb.log(flattened, step=state.iteration)
    return logger

def wandb_checkpoint(run=None, dir="checkpoints",
                format="epoch_{epoch}.ckpt"):
    import orbax.checkpoint as ocp

    run = run or wandb.run
    dir = Path(run.dir) / dir
    checkpointer = ocp.StandardCheckpointer()

    def log_fn(rng, state, **kwargs):
        vars = state.vars
        stats = state.metrics
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