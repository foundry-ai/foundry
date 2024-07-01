from stanza.data import Data, DataStream
from stanza.dataclasses import dataclass
from stanza.random import PRNGSequence
# import all reporting datatypes
from .reporting import *

from typing import (
    Any, TypeVar, Callable, Generic, Generator
)
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

import stanza.util

import time
import shutil
import os

import jax
import jax.numpy as jnp
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
            max_epochs: int,
            epoch_iterations: int,
            max_iterations: int,
            trace_dir: str | None,
            progress: Progress):
        self.rng_key = rng_key
        self.data = data
        self.max_epochs = max_epochs
        self.epoch_iterations = epoch_iterations
        self.max_iterations = max_iterations
        self.progress = progress
        self.trace_dir = trace_dir

        if self.progress is not None:
            self.iteration_task = progress.add_task("Iteration", total=max_iterations)
            self.epoch_task = progress.add_task("Epoch", total=max_epochs)
            self.epoch_iteration_task = progress.add_task("Epoch Iteration", total=epoch_iterations)
        else:
            self.iteration_task = None
            self.epoch_task = None
            self.epoch_iteration_task = None

    def epochs(self) -> "Generator[Epoch[Sample]]":
        iterations = 0
        if self.progress:
            self.progress.reset(self.epoch_task, total=self.max_epochs)
            self.progress.reset(self.iteration_task, total=self.max_iterations)
        rng = PRNGSequence(self.rng_key)
        try:
            for i in range(self.max_epochs):
                self.data = self.data.reset()
                epoch_iterations = min(self.epoch_iterations, self.max_iterations - iterations)
                yield Epoch(self, next(rng), i, iterations, epoch_iterations)
                iterations = iterations + epoch_iterations
                if self.progress:
                    self.progress.advance(self.epoch_task)
        finally:
            if self.progress:
                self.progress.refresh()
            if self.trace_dir is not None:
                jax.profiler.stop_trace()
                traces = list(self.trace_dir.glob("**/perfetto*"))
                if not traces:
                    logger.warning("No traces found!")
                else:
                    trace = traces[0]
                    dest = Path(os.environ.get("HOME")) / "traces" / (self.trace_dir.name + "_trace.json.gz")
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copyfile(trace, dest)

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
    
    def steps(self) -> "Generator[Step[Sample]]":
        prev_iterations = self.prev_iterations
        if self.loop.progress:
            self.loop.progress.reset(
                self.loop.epoch_iteration_task, total=self.epoch_iterations
            )
        rng = PRNGSequence(self.rng_key)
        for i in range(self.epoch_iterations):
            total_iter = prev_iterations + i
            if self.loop.trace_dir is not None and total_iter > 0:
                step_profile = jax.profiler.StepTraceAnnotation("step", step_num=total_iter)
            else:
                step_profile = nullcontext()

            with step_profile:
                with jax.profiler.TraceAnnotation("data_fetch"):
                    data, batch = self.loop.data.next()
                    self.loop.data = data
                with jax.profiler.TraceAnnotation("run_step"):
                    yield Step(batch, next(rng), self.num, 
                        i, total_iter)
            if self.loop.progress:
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
def loop(data : Data[Sample], *, batch_size, rng_key,
         epochs=None, iterations=None, progress=True,
         log_compiles=False, trace=False) -> Generator[Loop[Sample], None, None]:
    if (epochs is None and iterations is None) or \
            (epochs is not None and iterations is not None):
        raise ValueError("Must specify either iterations or epochs!")
    epoch_iterations = len(data) // batch_size
    if iterations is None:
        iterations = epochs*epoch_iterations
    if epochs is None:
        epochs = (iterations + epoch_iterations - 1) // epoch_iterations
    
    with data.stream(batch_size=batch_size) as stream:
        rng_key, shuffle_key = jax.random.split(rng_key)
        stream = stream.shuffle(shuffle_key)
        if progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=Style(color="green")),
                MofNColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            )
        else: progress = nullcontext()
        if log_compiles: compile_logger = jax.log_compiles()
        else: compile_logger = nullcontext()
        if trace:
            trace_dir = Path("/tmp/jax-traces") / time.strftime("%Y_%m_%d-%H_%M_%S")
            trace_dir.mkdir(exist_ok=True, parents=True)
        else:
            trace_dir = None
        loop = Loop(
            rng_key,
            stream,
            epochs, epoch_iterations,
            iterations,
            progress=progress,
            trace_dir=trace_dir
        )
        with progress, compile_logger:
            yield loop

@dataclass
class LossOutput:
    loss: ArrayLike = 0.
    metrics: Metrics = None
    var_updates: Vars = None

@partial(jax.jit, static_argnums=(0,))
def batched_loss(loss_fn, vars, rng_key, batch, **kwargs):
    loss = lambda rng, sample: loss_fn(vars, rng, sample, **kwargs)
    vmap_loss = jax.vmap(loss,
        in_axes=0,
        out_axes=LossOutput(
            var_updates=None,
            loss=0,
            metrics=0),
        axis_name="batch"
    )
    batch_size = stanza.util.axis_size(batch, 0)
    rng_keys = jax.random.split(rng_key, batch_size)

    output = vmap_loss(rng_keys, batch)

    stats = jax.tree_map(lambda x: jnp.mean(x, 0), output.metrics)
    loss = jnp.mean(output.loss)
    var_updates = output.var_updates
    return LossOutput(
        loss=loss,
        metrics=stats,
        var_updates=var_updates
    )

def batch_loss(loss_fn):
    return partial(batched_loss, loss_fn)

@partial(jax.profiler.annotate_function, name="update")
@partial(jax.jit, static_argnums=(0,1), donate_argnums=(2,3))
def step(batch_loss_fn : Callable[[Vars, jax.Array, Sample], LossOutput], 
        optimizer : optax.GradientTransformationExtraArgs, 
        opt_state : OptState, 
        vars : Vars, 
        rng_key : jax.Array,
        batch : Sample,
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
    return opt_state, vars, output.metrics

@partial(jax.profiler.annotate_function, name="eval")
@partial(jax.jit, static_argnums=(0,))
def eval(batch_loss_fn, vars: Vars, rng_key: jax.Array, batch: Sample):
    output = batch_loss_fn(vars, rng_key, batch)
    return output.metrics