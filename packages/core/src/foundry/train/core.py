from foundry.data import DataStream, StreamBuilder
from foundry.core import tree
from foundry.core.dataclasses import dataclass
from foundry.random import PRNGSequence
# import all reporting datatypes
from .reporting import *

from typing import (
    Any, TypeVar, Callable, Generic
)
from collections.abc import Iterator
from jax.typing import ArrayLike
from functools import partial
from contextlib import contextmanager, nullcontext
from pathlib import Path

from rich.text import Text as RichText
from rich.progress import (
    Progress, ProgressColumn,
    TextColumn, BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn
)
from rich.style import Style

import foundry.util
import itertools
import jax
import foundry.numpy as jnp
import optax # type: ignore

import logging
logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
OptState = Any
Vars = Any
Metrics = Any

class Loop(Generic[Sample]):
    def __init__(self,
            rng_key: jax.Array,
            data: DataStream[Sample],
            max_iterations: int,
            trace_dir: str | None,
            progress: Progress,
            show_epochs: bool):
        self.rng_key = rng_key
        self.data = data
        try: epoch_iterations = len(data)
        except TypeError: epoch_iterations = None
        self.epoch_iterations = epoch_iterations
        self.max_epochs = (max_iterations // epoch_iterations 
                           if epoch_iterations is not None else None)
        self.max_iterations = max_iterations
        self.progress = progress
        self.trace_dir = trace_dir
        self.show_epochs = (self.epoch_iterations is not None and self.max_iterations > self.epoch_iterations) and show_epochs

        if self.progress is not None:
            self.iteration_task = progress.add_task("Iteration", total=max_iterations)
            if self.show_epochs:
                self.epoch_task = progress.add_task("Epoch", total=self.max_epochs)
                self.epoch_iteration_task = progress.add_task("Epoch Iteration", total=epoch_iterations)
        else:
            self.iteration_task = None
            self.epoch_task = None
            self.epoch_iteration_task = None

    def epochs(self) -> "Iterator[Epoch[Sample]]":
        if self.progress:
            self.progress.reset(self.iteration_task, total=self.max_iterations)
            if self.show_epochs:
                self.progress.reset(self.epoch_task, total=self.max_epochs)
                self.progress.reset(self.epoch_iteration_task, total=self.epoch_iterations)

        iterations = 0
        rng = PRNGSequence(self.rng_key) if self.rng_key is not None else None
        try:
            for e in itertools.count():
                epoch_iterations = self.max_iterations - iterations
                if self.epoch_iterations is not None:
                    epoch_iterations = min(epoch_iterations, self.epoch_iterations)
                sk = next(rng) if rng is not None else None
                yield Epoch(self, sk, e, iterations, epoch_iterations)
                iterations = iterations + epoch_iterations
                if self.progress and self.show_epochs:
                    self.progress.advance(self.epoch_task)
                if iterations >= self.max_iterations:
                    break
        finally:
            if self.progress:
                self.progress.refresh()
            if self.trace_dir is not None:
                jax.profiler.stop_trace()

class Epoch(Generic[Sample]):
    def __init__(self, loop: Loop[Sample], rng_key: jax.Array,
                    epoch, prev_iterations, epoch_iterations):
        self.rng_key = rng_key
        self.loop = loop
        self.num = epoch
        self.prev_iterations = prev_iterations
        self.epoch_iterations = epoch_iterations

    @property
    def data(self):
        return self.loop.data
    
    def steps(self) -> "Iterator[Step[Sample]]":
        prev_iterations = self.prev_iterations
        if self.loop.progress and self.loop.show_epochs:
            self.loop.progress.reset(
                self.loop.epoch_iteration_task, total=self.epoch_iterations
            )
        rng = PRNGSequence(self.rng_key) if self.rng_key is not None else None
        for i in range(self.epoch_iterations):
            total_iter = prev_iterations + i
            with jax.profiler.StepTraceAnnotation("step", step_num=total_iter):
                with jax.profiler.TraceAnnotation("data_fetch"):
                    data = self.loop.data
                    if not data.has_next(): data = data.reset()
                    if not data.has_next(): raise ValueError("Unable to reset stream!")
                    data, batch = data.next()
                    self.loop.data = data
                    sk = next(rng) if rng is not None else None
                with jax.profiler.TraceAnnotation("run_step"):
                    yield Step(batch, sk, self.num, 
                        i, total_iter)

            if self.loop.progress:
                if self.loop.show_epochs:
                    self.loop.progress.advance(self.loop.epoch_iteration_task)
                self.loop.progress.advance(self.loop.iteration_task)

            if self.loop.trace_dir is not None and total_iter == 0:
                jax.profiler.start_trace(str(self.loop.trace_dir),
                                         create_perfetto_trace=True)

class Step(Generic[Sample]):
    def __init__(self, batch : Sample, rng_key: jax.Array, epoch, epoch_iteration, iteration):
        self.rng_key = rng_key
        self.batch = batch
        self.epoch = epoch
        self.epoch_iteration = epoch_iteration
        self.iteration = iteration
    
    @property
    def num(self):
        return self.iteration

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return RichText(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )

@contextmanager
def loop(data : StreamBuilder[Sample], *, iterations, rng_key=None, 
         progress=True, show_epochs=True,
         log_compiles=False, trace=False) -> Iterator[Loop[Sample]]:
    with data.build() as stream:
        if progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=Style(color="green")),
                MofNColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                refresh_per_second=1
            )
            progress_ctx = progress
        else: 
            progress = None
            progress_ctx = nullcontext()
        if log_compiles: compile_logger = jax.log_compiles()
        else: compile_logger = nullcontext()
        if trace:
            trace_dir = Path("/tmp/jax-traces")#  / time.strftime("%Y_%m_%d-%H_%M_%S")
            trace_dir.mkdir(exist_ok=True, parents=True)
        else:
            trace_dir = None
        loop = Loop(
            rng_key,
            stream,
            iterations,
            progress=progress,
            show_epochs=show_epochs,
            trace_dir=trace_dir
        )
        with progress_ctx, compile_logger:
            yield loop


@dataclass
class LossOutput:
    loss: ArrayLike = 0.
    metrics: Metrics = None
    var_updates: Vars = None

LossFn = Callable[[Vars, jax.Array, Sample], LossOutput]

@partial(jax.jit, static_argnums=(0,))
def batched_loss(loss_fn : LossFn, vars, rng_key, batch, **kwargs) -> LossFn:
    loss = lambda rng, sample: loss_fn(vars, rng, sample, **kwargs)
    vmap_loss = jax.vmap(loss,
        in_axes=0,
        out_axes=LossOutput(
            var_updates=None,
            loss=0,
            metrics=0),
        axis_name="batch"
    )
    batch_size = tree.axis_size(batch, 0)
    rng_keys = jax.random.split(rng_key, batch_size)

    output = vmap_loss(rng_keys, batch)

    stats = jax.tree_map(lambda x: jnp.mean(x, 0), output.metrics)
    loss = jnp.mean(output.loss) if output.loss is not None else None
    var_updates = output.var_updates
    return LossOutput(
        loss=loss,
        metrics=stats,
        var_updates=var_updates
    )

def batch_loss(loss_fn):
    return partial(batched_loss, loss_fn)

@partial(jax.profiler.annotate_function, name="step")
@partial(jax.jit, static_argnums=(0,1), static_argnames=("return_grad", "return_grad_norm"), donate_argnums=(2,3))
def step(batch_loss_fn : LossFn, 
        optimizer : optax.GradientTransformationExtraArgs, 
        opt_state : OptState, 
        vars : Vars, 
        rng_key : jax.Array,
        batch : Sample,
        *,
        return_grad=False,
        return_grad_norm=False,
        **kwargs : dict[str,Any]):
    def batch_loss(params, state):
        vars = {"params": params, **state}
        output = batch_loss_fn(vars, rng_key, batch, **kwargs)
        return output.loss, output
    params = vars["params"]
    state = {k: v for k, v in vars.items() if k != "params"}
    grad_fn = jax.grad(batch_loss, argnums=0, has_aux=True)
    grad_only_fn = lambda params, _: grad_fn(params, state)[0]
    grads, output =  grad_fn(params, state)
    updates, opt_state = optimizer.update(grads, opt_state, params, grad_fn=grad_only_fn)
    params = optax.apply_updates(params, updates)
    var_updates = output.var_updates if output.var_updates is not None else {}
    vars = {"params": params, **var_updates}
    if return_grad:
        return opt_state, vars, grads, output.metrics
    elif return_grad_norm:
        grad_norm = optax.tree_utils.tree_l2_norm(grads)
        return opt_state, vars, grad_norm, output.metrics
    else:
        return opt_state, vars, output.metrics

@partial(jax.profiler.annotate_function, name="eval")
@partial(jax.jit, static_argnums=(0,))
def eval(batch_loss_fn, vars: Vars, rng_key: jax.Array, batch: Sample):
    output = batch_loss_fn(vars, rng_key, batch)
    return output.metrics

def eval_stream(batch_loss_fn, vars: Vars, 
        rng_key: jax.Array, stream: DataStream[Sample], batches=None):
    outputs = []
    r = PRNGSequence(rng_key)
    iter = range(batches) if batches is not None else itertools.count()
    if batches is None and not stream.has_next():
        stream = stream.reset()
    for _ in iter:
        if not stream.has_next():
            if batches is not None: stream = stream.reset()
            else: break
        stream, batch = stream.next()
        outputs.append(eval(batch_loss_fn, vars, next(r), batch))
    output = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x), 0), *outputs)
    return stream, output