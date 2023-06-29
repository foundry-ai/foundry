from stanza import Partial
from stanza.dataclasses import dataclass, field, replace
from stanza.util.logging import logger
from stanza.data import PyTreeData

from jax.random import PRNGKey

from typing import Any, List
from collections import namedtuple
from functools import partial

import stanza
import optax
import jax
import jax.numpy as jnp

@dataclass(jax=True)
class TrainState:
    epoch: int
    max_epoch: int
    total_iteration: int
    epoch_iteration: int
    max_iteration: int

    batch_iterator: Any
    last_stats : Any
    hook_states : List[Any]

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
    fn_state, loss, stats = batch_loss_fn(fn_params, fn_state,
                                            rng_key, batch)
    loss = jnp.mean(loss)
    stats = jax.tree_util.tree_map(jnp.mean, stats)
    return loss, (stats, fn_state)

@jax.jit
def _run_hooks(state, hooks):
    new_hook_states = []
    for h, hs in zip(hooks, state.hook_states):
        hs, state = h(hs, state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state

@dataclass(jax=True, kw_only=True)
class Trainer:
    optimizer: optax.GradientTransformation = optax.adam(1e-3)

    batch_size: int = field(default=32, jax_static=True)
    # optional, if not set
    # must be passed into train()
    epochs : int = field(default=None)
    max_iterations : int = field(default=None)

    @jax.jit
    def train_step(self, state, *,
                        loss_fn, batch_dataset):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        from stanza.util import shape_tree
        batch_data = batch_dataset.get(state.batch_iterator)
        batch_iterator = batch_dataset.next(state.batch_iterator)
        # Specify a buffer size!
        batch = PyTreeData.from_data(batch_data,
                            buffer_size=self.batch_size)
        batch_fn = Partial(_batch_loss_fn, loss_fn, state.fn_state, sk, 
                                batch.data)
        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, 
                                    state.opt_state, state.fn_params)
        fn_params = optax.apply_updates(state.fn_params, updates)

        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            total_iteration=state.total_iteration + 1,
            batch_iterator=batch_iterator,
            last_stats=stats,
            rng_key=rng_key,
            fn_params=fn_params, fn_state=fn_state, opt_state=opt_state)
        return state

    @jax.jit
    def _step_with_hooks(self, state, loss_fn, batch_dataset, hooks):
        # get rid of the hook state 
        # to avoid recompiling the training step
        hook_states = state.hook_states
        state = replace(state, hook_states=None)
        state = self.train_step(state,
            loss_fn=loss_fn, batch_dataset=batch_dataset)
        # put in the hook state, call the hooks
        # and get rid of the last stats
        state = replace(state, hook_states=hook_states)
        state = _run_hooks(state, hooks)
        state = replace(state, last_stats=None)
        return state

    def train_epoch(self, state, *,
                        loss_fn, dataset, hooks, jit=True):
        logger.trace("Tracing epoch step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)
        dataset = dataset.shuffle(sk)
        batch_dataset = dataset.batch(self.batch_size)

        step_fn = Partial(type(self)._step_with_hooks,
                    self,
                    loss_fn=loss_fn,
                    batch_dataset=batch_dataset,
                    hooks=hooks)
        # call the hooks before each epoch
        # step_state = _run_hooks(first_step_state, hooks)
        # run first step separately
        step_state = replace(state, batch_iterator=batch_dataset.start)
        cond_fn = lambda s: jnp.logical_and(s.total_iteration < s.max_iteration,
                                jnp.logical_not(batch_dataset.is_end(s.batch_iterator)))
        if jit:
            step_state = step_fn(step_state)
            step_state = jax.lax.while_loop(cond_fn, step_fn, step_state)
        else:
            while cond_fn(step_state):
                step_state = step_fn(step_state)

        return replace(state,
            epoch=state.epoch+1,
            epoch_iteration=0,
            total_iteration=step_state.total_iteration,
            hook_states=step_state.hook_states,
            rng_key=step_state.rng_key,
            fn_params=step_state.fn_params,
            fn_state=step_state.fn_state,
            opt_state=step_state.opt_state
        )

    def _train_loop(self, loss_fn, dataset, state, hooks, jit=True):
        logger.trace("Tracing training", only_tracing=True)
        # do the first epoch by hand (a) to handle
        # the first hook states and (b) to make debugging easier
        if jit:
            train_epoch = jax.jit(type(self).train_epoch)
            epoch_fn = Partial(train_epoch, self,
                            loss_fn=loss_fn, dataset=dataset,
                            hooks=hooks)
            state = epoch_fn(state)
            state = jax.lax.while_loop(
                lambda s: s.total_iteration < s.max_iteration,
                epoch_fn, state
            )
        else:
            epoch_fn = Partial(self.train_epoch,
                        loss_fn=loss_fn, dataset=dataset,
                        hooks=hooks, jit=False)
            while state.total_iteration < state.max_iteration:
                state = epoch_fn(state)
        # Run the hooks after finishing
        state = _run_hooks(state, hooks)
        logger.trace("Done tracing training", only_tracing=True)
        return state
    
    def train(self, loss_fn, dataset, rng_key,
                init_fn_params, init_fn_state=None,
                *,
                init_opt_state=None,
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
        if max_iterations is None and epochs is None:
            raise ValueError("Must specify either number of epochs or iterations")
        
        if max_iterations is None:
            num_batches = dataset.length // self.batch_size
            max_iterations = num_batches*epochs

        state = TrainState(
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
            opt_state=init_opt_state
        )
        train_fn = jax.jit(type(self)._train_loop) if jit else \
                    partial(type(self)._train_loop,jit=False)
        state = train_fn(self, loss_fn, dataset, state, hooks)
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