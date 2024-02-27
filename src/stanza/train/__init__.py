import stanza.struct as struct

from stanza.util.random import PRNGSequence
from stanza.util import MofNColumn, dict_flatten
from stanza.data import Data, DataLoader

from typing import Any, TypeVar, Callable
from functools import partial
from bdb import BdbQuit
from rich.progress import (
    Progress, TextColumn, BarColumn, 
    TimeRemainingColumn, TimeElapsedColumn
)
from rich.style import Style

from jax.typing import ArrayLike

import jax.numpy as jnp
import functools

import jax
import os
import optax # type: ignore

import logging
logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
OptState = Any
Vars = Any
Stats = Any

# Training hooks
@struct.dataclass
class TrainState:
    max_iterations: int
    max_epochs: int
    iterations_per_epoch: int

    iteration: int
    epoch: int
    epoch_iteration: int
    opt_state: OptState
    vars: Vars
    last_stats: Stats

@struct.dataclass
class LossOutput:
    loss: ArrayLike
    stats: Stats = None
    var_updates: Vars = None

def get_batch_size(batch):
    import numpy as np
    ns = np.array([jnp.shape(x)[0] for x in jax.tree_util.tree_leaves(batch)])
    assert np.all(ns == ns[0])
    return ns[0]

# makes a loss_fn into a batch_loss_fn
def batch_loss(loss_fn):
    @jax.jit
    def batched_loss(vars, i, rng_key, batch):
        vmap_loss = jax.vmap(loss_fn,
            in_axes=(None, None, 0, 0),
            out_axes=LossOutput(
                var_updates=None,
                loss=0,
                stats=0),
            axis_name="batch"
        )
        batch_size = get_batch_size(batch)
        rng_batch = jax.random.split(rng_key, batch_size)
        output = vmap_loss(vars, i,
            rng_batch, batch)
        var_updates = output.var_updates
        stats = jax.tree_map(lambda x: jnp.mean(x, 0), output.stats)
        loss = jnp.mean(output.loss)
        return LossOutput(loss=loss, stats=stats, var_updates=var_updates)
    return batched_loss

@partial(jax.jit, static_argnums=(0,1))
def _update(loss_fn, optimizer, 
            opt_state, vars, iteration, rng, batch):
    def batch_loss(params, state):
        vars = {"params": params, **state}
        output = loss_fn(vars, iteration, rng, batch)
        return output.loss, output
    params = vars["params"]
    state = {k: v for k, v in vars.items() if k != "params"}
    grad_fn = jax.grad(batch_loss, argnums=0, has_aux=True)
    grads, output =  grad_fn(params, state)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    var_updates = output.var_updates if output.var_updates is not None else {}
    vars = {"params": params, **var_updates}
    return opt_state, vars, output.stats

def fit(*, data : Data[Sample],
        optimizer : optax.GradientTransformation,
        batch_loss_fn : Callable[[Vars, jax.Array, jax.Array, Sample], LossOutput],
        init_vars : Vars, 
        rng_key : jax.Array,
        max_epochs : int = None,
        max_iterations : int = None,
        batch_size : int,
        hooks=[],
        trace_dir=None):
    """A fit a model to data using a
    given optimizer and loss function.
    """
    if max_epochs is None and max_iterations is None:
        raise ValueError("max_epochs or max_iterations must be specified")
    batch_loss_fn = jax.jit(batch_loss_fn)
    rng = PRNGSequence(rng_key)
    dataloader = DataLoader(
        data, batch_size=batch_size,
        rng_key=next(rng), shuffle=True,
        drop_jagged=True
    )
    iteration = 0
    iterations_per_epoch = len(dataloader)
    if max_iterations is None:
        max_iterations = max_epochs * iterations_per_epoch
    if max_epochs is None:
        max_epochs = (max_iterations + iterations_per_epoch - 1) // iterations_per_epoch
    vars = init_vars
    opt_state = optimizer.init(vars["params"])

    pbar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=Style(color="green")),
                MofNColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            )
    total_iteration_task = pbar.add_task("Iteration", total=max_iterations)
    epoch_task = pbar.add_task("Epoch", total=max_epochs)
    iteration_task = pbar.add_task("Epoch Iteration", total=iterations_per_epoch)

    if trace_dir is not None:
        logger.info(f"Saving trace to: [blue]{trace_dir}[/blue]")
        jax.profiler.start_trace(trace_dir, create_perfetto_trace=True)
    try:
        with pbar:
            for epoch in range(max_epochs):
                pbar.reset(iteration_task)
                for i, batch in enumerate(dataloader):
                    opt_state, vars, stats = _update(
                        batch_loss_fn, optimizer,
                        opt_state, vars,
                        iteration, next(rng), batch
                    )
                    state = TrainState(
                        max_iterations, max_epochs,
                        iterations_per_epoch,
                        iteration, epoch, i,
                        opt_state, vars,
                        stats
                    )
                    for h in hooks:
                        h(next(rng), state)
                    iteration += 1
                    pbar.update(iteration_task, completed=i+1)
                    pbar.update(total_iteration_task, completed=iteration)
                    if iteration >= max_iterations:
                        break
                pbar.update(epoch_task, advance=1)
            # do one final hook call
            state = TrainState(
                max_iterations, max_epochs,
                iterations_per_epoch,
                iteration,
                epoch + 1, 0,
                opt_state, vars,
                stats
            )
            for h in hooks:
                h(next(rng), state)
    except (KeyboardInterrupt, BdbQuit):
        # Hard-kill wandb process on manual exit
        cmd = "ps aux|grep wandb|grep -v grep | awk '\''{print $2}'\''|xargs kill -9"
        os.system(cmd)
        raise KeyboardInterrupt
    finally:
        if trace_dir is not None:
            jax.profiler.stop_trace()
    return vars

# Hook decorators
def every_n_iterations(*hooks, n=1):
    if not hooks:
        return partial(every_n_iterations, n=n)
    def wrapped(rng_key, state, **kwargs):
        if state.epoch % n == 0 and \
                state.epoch_iteration + 1 == state.iterations_per_epoch:
            for h in hooks:
                h(rng_key, state, **kwargs)
    if len(hooks) == 1:
        wrapped = functools.wraps(hooks[0])(wrapped)
    return wrapped

def every_n_epochs(*hooks, n=1):
    if not hooks:
        return partial(every_n_epochs, n=n)
    def wrapped(rng_key, state, **kwargs):
        if state.epoch % n == 0 and \
                state.epoch_iteration == 0:
            for h in hooks:
                h(rng_key, state, **kwargs)
    if len(hooks) == 1:
        wrapped = functools.wraps(hooks[0])(wrapped)
    return wrapped

every_epoch = partial(every_n_epochs, n=1)

# console logging hooks
def console_logger(prefix=None, suffix=None):
    def log_fn(rng_key, state, **kwargs):
        flattened = dict_flatten(
            state.last_stats,
            prefix=prefix, suffix=suffix
        )
        p = f"epoch: {state.epoch:>3} iter: {state.iteration:>4}"
        for k, v in flattened.items():
            logger.info(f"{p} - {k}: {v}")
    return log_fn

# validation hook
def validate(*, hooks, 
            dataset, batch_loss_fn, 
            batch_size
        ):
    # the validation dataloader
    dataloader = DataLoader(dataset,
        batch_size=batch_size, shuffle=False
    )
    def hook_fn(rng_key, state, **kwargs):
        all_stats = []
        rng = PRNGSequence(rng_key)
        for batch in dataloader:
            output = batch_loss_fn(
                state.vars, state.iteration,
                next(rng), batch
            )
            all_stats.append(output.stats)
        all_stats = jax.tree_map(lambda *x: jnp.stack(x, 0), *all_stats)
        all_stats = jax.tree_map(lambda x: jnp.mean(x, 0), all_stats)

        state = struct.replace(state, last_stats=all_stats)
        for h in hooks:
            h(next(rng), state)
    return hook_fn
