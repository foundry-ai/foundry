from stanza import Partial
from stanza.dataclasses import dataclass, field, replace
from stanza.util.logging import logger
from stanza.util import LoopState, run_hooks, init_hooks as _init_hooks
from stanza.data import Data, Iterator, PyTreeData

from jax.random import PRNGKey

from typing import Any, List, Callable
from collections import namedtuple
from functools import partial

import stanza
import stanza.util
import optax
import jax
import jax.numpy as jnp

@dataclass(jax=True)
class TrainState(LoopState):
    epoch: int
    max_epochs: int
    epoch_iteration: int

    loss_fn: Callable

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any

@dataclass(jax=True)
class TrainResults:
    fn_params: Any
    fn_state: Any
    opt_state: Any
    hook_states: List[Any]

@jax.jit
def _batch_loss_fn(loss_fn, fn_state,
                   rng_key, batch, fn_params):
    logger.trace("Tracing batch loss", only_tracing=True)
    batch_loss_fn = jax.vmap(loss_fn,
                    in_axes=(None, None, None, 0),
                    axis_name='batch')
    fn_state, loss, stats = batch_loss_fn(fn_state, fn_params,
                                            rng_key, batch)
    loss = jnp.mean(loss)
    stats = jax.tree_map(jnp.mean, stats)
    return loss, (stats, fn_state)

@dataclass(jax=True, kw_only=True)
class Trainer:
    optimizer: optax.GradientTransformation = optax.adam(1e-3)
    batch_size: int = field(default=32, jax_static=True)
    # optional, if not set
    # must be passed into train()
    epochs : int = field(default=None)
    max_iterations : int = field(default=None)

    @jax.jit
    def train_step(self, state, batch):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch = PyTreeData.from_data(batch,
                    buffer_size=self.batch_size)
        batch_fn = Partial(_batch_loss_fn, 
                state.loss_fn, state.fn_state, 
                sk, batch.data)
        batch_grad_fn = jax.grad(batch_fn, has_aux=True)
        grads, (stats, fn_state) = batch_grad_fn(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, 
                            state.opt_state, state.fn_params)
        fn_params = optax.apply_updates(state.fn_params, updates)
        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            iteration=state.iteration+ 1,
            last_stats=stats, rng_key=rng_key,
            fn_params=fn_params, 
            fn_state=fn_state, opt_state=opt_state)
        return state
    
    def train_step_with_hooks(self, state, batch):
        state = self.train_step(state, batch)
        state = run_hooks(state)
        return state

    def train_epoch(self, state, dataset, *, jit=True):
        logger.trace("Tracing epoch step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)

        shuffled_dataset = dataset.shuffle(sk)
        batch_dataset = shuffled_dataset.batch(self.batch_size)

        step_fn = Partial(type(self).train_step_with_hooks, self)
        # call the hooks before each epoch
        # run first step separately
        state = run_hooks(state)
        state = batch_dataset.scan(step_fn, state, jit=jit,
                limit=state.max_iterations - state.iteration)
        return replace(state,
            epoch=state.epoch+1,
            epoch_iteration=0,
        )

    def run(self, state, dataset, jit=True):
        logger.trace("Tracing training", only_tracing=True)
        if jit:
            train_epoch = jax.jit(type(self).train_epoch)
            epoch_fn = Partial(train_epoch, self,
                               dataset=dataset)
        else:
            epoch_fn = Partial(self.train_epoch,
                               dataset=dataset, jit=False)
        # populate the last_stats if necessary
        state = stanza.util.loop(epoch_fn, state, jit=jit)
        state = run_hooks(state)
        logger.trace("Done tracing training", only_tracing=True)
        return state
    
    def init(self, loss_fn, data_sample, max_iterations, rng_key, 
             init_fn_params, init_fn_state=None,
             *,
             init_opt_state=None, epochs=None, hooks=[],
             init_hooks=True):
        epochs = self.epochs if epochs is None else epochs
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        if init_opt_state is None:
            init_opt_state = self.optimizer.init(init_fn_params)
        assert max_iterations is not None

        _, _, stats = loss_fn(init_fn_state, init_fn_params, 
                            PRNGKey(42), data_sample)
        stats = jax.tree_map(jnp.mean, stats)
        stats = jax.tree_map(jnp.zeros_like, stats)

        state = TrainState(
            iteration=0,
            max_iterations=max_iterations,
            hooks=hooks,
            hook_states=[None]*len(hooks),

            epoch=0, 
            max_epochs=epochs,
            epoch_iteration=0,

            loss_fn=loss_fn,
            last_stats=stats,

            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state
        )
        if init_hooks:
            state = _init_hooks(state)
        return state
    
    def train(self, loss_fn, dataset, *args,
                epochs=None, max_iterations=None, jit=True, **kwargs):
        epochs = self.epochs if epochs is None else epochs
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        if max_iterations is None and epochs is not None:
            max_iterations = len(dataset) * epochs // self.batch_size

        state = self.init(loss_fn, dataset[0], max_iterations, *args, epochs=epochs, **kwargs)
        run = jax.jit(type(self).run) if jit else \
                    partial(type(self).run, jit=False)
        state = run(self, state, dataset)
        results = TrainResults(
            fn_params = state.fn_params,
            fn_state = state.fn_state,
            opt_state = state.opt_state,
            hook_states = state.hook_states
        )
        return results

@dataclass(jax=True)
class SAMTrainState(TrainState):
    sub_opt_state: Any

@dataclass(jax=True)
class SAMTrainResults(TrainResults):
    sub_opt_state: Any

@dataclass(jax=True)
class SAMTrainer(Trainer):
    sub_optimizer: optax.GradientTransformation = optax.sgd(1e-5)

    @jax.jit
    def train_step(self, state, *,
                        loss_fn, batch_dataset):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch_data = batch_dataset.get(state.batch_iterator)
        batch_iterator = batch_dataset.next(state.batch_iterator)
        # Specify a buffer size!
        batch = PyTreeData.from_data(batch_data,
                            buffer_size=self.batch_size)
        batch_fn = Partial(_batch_loss_fn, loss_fn, state.fn_state, sk, 
                                batch.data)
        batch_fn_grad = jax.grad(batch_fn, has_aux=True)
        grads, (stats, fn_state) = batch_fn_grad(state.fn_params)
        # take the sub-step
        sub_updates, sub_opt_state = self.sub_optimizer.update(grads, 
                                    state.sub_opt_state, state.fn_params)
        sub_params = optax.apply_updates(state.fn_params, sub_updates)

        grads, (stats, fn_state) = batch_fn_grad(sub_params)
        updates, opt_state = self.optimizer.update(grads, 
                                    state.opt_state, state.fn_params)
        fn_params = optax.apply_updates(state.fn_params, updates)

        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            total_iteration=state.total_iteration + 1,
            batch_iterator=batch_iterator,
            last_stats=stats,
            rng_key=rng_key,
            fn_params=fn_params, fn_state=fn_state, 
            opt_state=opt_state, sub_opt_state=sub_opt_state)
        return state

    def train(self, loss_fn, dataset, rng_key,
                init_fn_params, init_fn_state=None,
                *,
                init_opt_state=None,
                init_sub_opt_state=None,
                # can specify epochs either through
                # constructor or override in train() 
                # function
                epochs=None, max_iterations=None, hooks=[],
                jit=True, **kwargs):
        # epochs and max_iterations can come from either
        # the trainer parameters or the train parameters
        epochs = self.epochs if epochs is None else epochs
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        if init_opt_state is None:
            init_opt_state = self.optimizer.init(init_fn_params)
        if init_sub_opt_state is None:
            init_sub_opt_state = self.optimizer.init(init_fn_params)
        if max_iterations is None and epochs is None:
            raise ValueError("Must specify either number of epochs or iterations")
        
        if max_iterations is None:
            num_batches = (dataset.length - 1) // self.batch_size + 1
            max_iterations = num_batches*epochs

        state = SAMTrainState(
            epoch=0, 
            max_epoch=epochs,
            epoch_iteration=0,
            total_iteration=0,
            max_iteration=max_iterations,

            # Used at the step level
            # not the state level
            batch_iterator=None,

            last_stats=None,
            hook_states=[None]*len(hooks),

            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state,
            sub_opt_state=init_sub_opt_state
        )

        train_fn = jax.jit(type(self)._train_loop) if jit else \
                    partial(type(self)._train_loop, jit=False)
        state = train_fn(self, 
            stanza.Partial(loss_fn), dataset, state, hooks)

        results = SAMTrainResults(
            fn_params = state.fn_params,
            fn_state = state.fn_state,
            opt_state = state.opt_state,
            sub_opt_state = state.sub_opt_state,
            hook_states = state.hook_states
        )
        return results