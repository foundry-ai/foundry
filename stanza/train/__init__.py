from stanza import Partial
from stanza.dataclasses import dataclass, field, replace, unpack
from stanza.util.logging import logger
from stanza.util.loop import LoopState, run_hooks, init_hooks as _init_hooks, loop
from stanza.data import Data, Iterator, PyTreeData

from jax.random import PRNGKey

from typing import Any, List, Callable
from collections import namedtuple
from functools import partial

import chex
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

    loss_fn: Callable #takes in (fn_fn_state, fn_params, rng_key, batch)

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
def _batch_loss_fn(loss_fn, fn_state, fn_params,
                        rng_key, batch):
    logger.trace("Tracing batch loss", only_tracing=True)
    batch_loss_fn = jax.vmap(loss_fn,
                    in_axes=(None, None, None, 0),
                    out_axes=(None, 0, 0),
                    axis_name='batch')
    fn_state, loss, stats = batch_loss_fn(fn_state, fn_params,
                                            rng_key, batch)
    loss = jnp.mean(loss)
    stats = jax.tree_map(jnp.mean, stats)
    return fn_state, loss, stats

def batch_loss(loss_fn):
    return Partial(_batch_loss_fn, Partial(loss_fn))

def _trainer_loss_fn(loss_fn, fn_state, rng_key, batch, fn_params):
    fn_state, loss, stats = loss_fn(fn_state, fn_params, rng_key, batch)
    return loss, (fn_state, stats)

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
        batch_fn = Partial(_trainer_loss_fn, 
                state.loss_fn, state.fn_state, 
                sk, batch.data)
        batch_grad_fn = jax.grad(batch_fn, has_aux=True)
        grads, (fn_state, stats) = batch_grad_fn(state.fn_params)

        if fn_state is not None or state.fn_state is not None:
            chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)

        updates, opt_state = self.optimizer.update(grads, 
                            state.opt_state, state.fn_params)
        fn_params = optax.apply_updates(state.fn_params, updates)
        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            iteration=state.iteration+ 1,
            last_stats=replace(
                state.last_stats,
                train=stats
            ), rng_key=rng_key,
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
        state = loop(epoch_fn, state, jit=jit)
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

        batch_sample = jax.tree_map(
            lambda x: jnp.repeat(jnp.array(x)[jnp.newaxis, ...], self.batch_size, axis=0),
            data_sample
        )
        _, _, stats = loss_fn(init_fn_state, init_fn_params, 
                            PRNGKey(42), batch_sample)
        stats = jax.tree_map(jnp.zeros_like, stats)

        state = TrainState(
            iteration=0,
            max_iterations=max_iterations,
            hooks=hooks,
            hook_states=None,

            epoch=0, 
            max_epochs=epochs,
            epoch_iteration=0,

            loss_fn=loss_fn,
            last_stats={"train":stats},

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
            iter_per_epoch = len(dataset) // self.batch_size
            max_iterations = iter_per_epoch * epochs
        if max_iterations is None:
            raise RuntimeError("Either epochs or max_iterations must be set")

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
    normalize: bool = field(default=True, jax_static=True)
    # Number of iterations to run SAM for
    sam_iterations: int = field(default=None)
    resample: bool = field(default=False, jax_static=True)

    @jax.jit
    def train_step(self, state, batch):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch = PyTreeData.from_data(batch,
                    buffer_size=self.batch_size)
        if self.resample:
            sam_batch = Data.from_pytree(
                jax.tree_map(lambda x: x[:self.batch_size//2], batch.data),
            )
            batch = Data.from_pytree(
                jax.tree_map(lambda x: x[self.batch_size//2:], batch.data),
            )
            gen_batch_fn = Partial(_trainer_loss_fn, 
                    state.loss_fn, state.fn_state, 
                    sk)
            gen_batch_grad_fn = jax.grad(gen_batch_fn, has_aux=True, argnums=1)
            batch_grad_fn = Partial(gen_batch_grad_fn, batch.data)
            sam_batch_grad_fn = Partial(gen_batch_grad_fn, sam_batch.data)
        else:
            batch_fn = Partial(_trainer_loss_fn, 
                    state.loss_fn, state.fn_state, 
                    sk, batch.data)
            batch_grad_fn = jax.grad(batch_fn, has_aux=True)
            sam_batch_grad_fn = batch_grad_fn

        def sam_update(state):
            grads, (_, _) = sam_batch_grad_fn(state.fn_params)
            if self.normalize:
                global_norm = optax.global_norm(grads)
                grads = jax.tree_map(lambda x: x / global_norm, grads)
            updates, sub_opt_state = self.sub_optimizer.update(grads, 
                                state.sub_opt_state, state.fn_params)
            sub_fn_params = optax.apply_updates(state.fn_params, updates)
            grads, (fn_state, stats) = batch_grad_fn(sub_fn_params)
            chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)
            updates, opt_state = self.optimizer.update(grads, 
                                state.opt_state, state.fn_params)
            fn_params = optax.apply_updates(state.fn_params, updates)
            state = replace(state,
                epoch_iteration=state.epoch_iteration + 1,
                iteration=state.iteration + 1,
                last_stats=stats, rng_key=rng_key,
                fn_params=fn_params, 
                fn_state=fn_state,
                opt_state=opt_state, sub_opt_state=sub_opt_state)
            return state

        def regular_update(state):
            grads, (fn_state, stats) = batch_grad_fn(state.fn_params)

            if fn_state is not None or state.fn_state is not None:
                chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)

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
        if self.sam_iterations is None:
            state = sam_update(state)
        else:
            state = jax.lax.cond(state.iteration < self.sam_iterations,sam_update,regular_update,state)
        return state

    def init(self, *args, init_sub_opt_state=None, **kwargs):
        state = super().init(*args, **kwargs)
        if init_sub_opt_state is None:
            init_sub_opt_state = self.sub_optimizer.init(state.fn_params)
        state = unpack(state)
        state = SAMTrainState(**state,
                      sub_opt_state=init_sub_opt_state)
        return state