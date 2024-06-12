import stanza.struct as struct

from stanza.random import PRNGSequence
from stanza.util import MofNColumn
from stanza.data import Data, DataLoader

# import all reporting datatypes
from .reporting import *

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
import itertools

import jax
import optax # type: ignore

import logging
logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
OptState = Any
Vars = Any
Metrics = Any

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
    metrics: Metrics

@struct.dataclass
class LossOutput:
    loss: ArrayLike = 0.
    metrics: Metrics = None
    var_updates: Vars = None

def get_batch_size(batch):
    import numpy as np
    ns = np.array([jnp.shape(x)[0] for x in jax.tree_util.tree_leaves(batch)])
    assert np.all(ns == ns[0])
    return ns[0]

@struct.dataclass
class IterationInfo:
    iteration: int
    epoch: int
    max_iteration: int
    max_epoch: int

# makes a loss_fn into a batch_loss_fn
def batch_loss(loss_fn):
    @jax.jit
    def batched_loss(vars, i, rng_key, batch):
        vmap_loss = jax.vmap(loss_fn,
            in_axes=(None, None, 0, 0),
            out_axes=LossOutput(
                var_updates=None,
                loss=0,
                metrics=0),
            axis_name="batch"
        )
        batch_size = get_batch_size(batch)
        rng_batch = jax.random.split(rng_key, batch_size)
        output = vmap_loss(vars, i,
            rng_batch, batch)
        var_updates = output.var_updates
        stats = jax.tree_map(lambda x: jnp.mean(x, 0), output.metrics)
        loss = jnp.mean(output.loss)
        return LossOutput(loss=loss, metrics=stats, var_updates=var_updates)
    return batched_loss

@partial(jax.jit, static_argnums=(0,1), donate_argnums=(2,3))
def _update(loss_fn, optimizer, 
            opt_state, vars, epoch, iteration,
            max_epoch, max_iteration, rng, batch):
    def batch_loss(params, state):
        vars = {"params": params, **state}
        info = IterationInfo(iteration, epoch, max_iteration, max_epoch)
        output = loss_fn(vars, info, rng, batch)
        return output.loss, output
    params = vars["params"]
    state = {k: v for k, v in vars.items() if k != "params"}
    grad_fn = jax.grad(batch_loss, argnums=0, has_aux=True)
    grad_only_fn = lambda params, _: grad_fn(params, state)[0]
    grads, output =  grad_fn(params, state)
    # grad_fn allows the use of the "sam" optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params, grad_fn=grad_only_fn)
    params = optax.apply_updates(params, updates)
    var_updates = output.var_updates if output.var_updates is not None else {}
    vars = {"params": params, **var_updates}
    return opt_state, vars, output.metrics

def fit(*, data : Data[Sample],
        rng_key : jax.Array,
        optimizer : optax.GradientTransformation,
        batch_loss_fn : Callable[[Vars, IterationInfo, jax.Array, Sample], LossOutput],
        init_vars : Vars, 
        init_opt_state : OptState = None,
        donate_init_vars : bool = False,
        donate_init_opt_state : bool = False,
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
    # recompute max_epochs based off of max_iterations
    max_epochs = (max_iterations + iterations_per_epoch - 1) // iterations_per_epoch

    vars = init_vars if donate_init_vars else jax.tree_map(lambda x: jnp.copy(x), init_vars)
    optimizer = optax.with_extra_args_support(optimizer)
    if init_opt_state is None:
        opt_state = optimizer.init(vars["params"])
    else:
        opt_state = init_opt_state if donate_init_opt_state else jax.tree_map(lambda x: jnp.copy(x), init_opt_state)
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
                        epoch, iteration, 
                        max_epochs, max_iterations,
                        next(rng), batch
                    )
                    state = TrainState(
                        max_iterations, max_epochs,
                        iterations_per_epoch,
                        iteration, epoch, i,
                        opt_state, vars,
                        stats
                    )
                    for h in hooks:
                        h(rng, state)
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
                h(rng, state)
    finally:
        if trace_dir is not None:
            jax.profiler.stop_trace()
    return vars

# Hook decorators
def every_n_iterations(n, *hooks):
    if not hooks:
        return partial(every_n_iterations, n=n)
    def wrapped(rng, state, **kwargs):
        if state.iteration % n == 0:
            for h in hooks:
                h(rng, state, **kwargs)
    if len(hooks) == 1:
        wrapped = functools.wraps(hooks[0])(wrapped)
    return wrapped

def every_n_epochs(n, *hooks):
    if not hooks:
        return partial(every_n_epochs, n=n)
    def wrapped(rng, state, **kwargs):
        if state.epoch % n == 0 and \
                state.epoch_iteration == 0:
            for h in hooks:
                h(rng, state, **kwargs)
    if len(hooks) == 1:
        wrapped = functools.wraps(hooks[0])(wrapped)
    return wrapped

every_epoch = partial(every_n_epochs, 1)

# console logging hooks
def console_logger(*data_hooks, logger=None, prefix=None, suffix=None, metrics=False):
    logger = logger if logger is not None else logging.getLogger(__name__)
    def log_hook(rng, state, *, log=None, **kwargs):
        r = []
        if log is not None:
            r.append(log)
        if metrics:
            r.append(state.metrics)
        for hook in data_hooks:
            r.append(hook(rng, state, **kwargs))

        flattened = dict_flatten(*r,
            prefix=prefix, suffix=suffix
        )
        p = f"epoch: {state.epoch:>3} iter: {state.iteration:>4}"
        for k, v in flattened.items():
            logger.info(f"{p} - {k}: {v}")
    return log_hook

def log_to(data_hook, *log_hooks):
    def hook_fn(rng, state, log=None, **kwargs):
        extra_log = data_hook(rng, state, **kwargs) if data_hook is not None else None
        log = log, extra_log if log is not None else extra_log
        for h in log_hooks:
            h(rng, state, log=log)
    return hook_fn

# validation hook
def validate(*hooks,
            data, 
            batch_loss_fn,
            batch_size=None,
            rng_key=None, # the rng key to use for batches
            batches=None, # if we should use a fixed number of batches
            log_hooks=[]
        ):
    if log_hooks:
        hooks = hooks + tuple(log_hooks)

    # the validation dataloader
    # note that this is shared across hook calls

    if batches is not None:
        if rng_key is None:
            raise ValueError("rng_key must be specified if batches is specified")
        # if we are using random test
        # batches
        dataloader = DataLoader(data,
            batch_size=batch_size, shuffle=True,
            drop_jagged=True, rng_key=rng_key
        )
        dataloader_cycle = dataloader.cycle()
        def iter_batches():
            return itertools.islice(dataloader_cycle, batches)
    else:
        # otehrwise go through
        # the entire dataset
        dataloader = DataLoader(data,
            batch_size=batch_size, shuffle=False,
        )
        def iter_batches():
            return iter(dataloader)

    @jax.jit
    def metric_fn(vars: Vars, iteration: int, rng_key: jax.Array, batch: Sample):
        output = batch_loss_fn(vars, iteration, rng_key, batch)
        return output.metrics

    def hook_fn(rng, state, *, log=None, **kwargs):
        all_stats = []
        for batch in iter_batches():
            metrics = metric_fn(
                state.vars, state.iteration,
                next(rng), batch
            )
            all_stats.append(metrics)
        all_stats = jax.tree_util.tree_map(lambda *x: jnp.stack(x, 0), *all_stats)
        all_stats = jax.tree_util.tree_map(lambda x: jnp.mean(x, 0), all_stats)
        for h in hooks:
            h(rng, state, log=all_stats, **kwargs)
    return hook_fn
