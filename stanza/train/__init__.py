from typing import Callable, Any
from stanza.util.logging import logger, pbar
from stanza.util.dataclasses import dataclass, field
from jax.random import PRNGKey
from stanza import Partial
from functools import partial

from collections import namedtuple

import optax
import jax
import jax.numpy as jnp

@dataclass(jax=True)
class EpochState:
    epoch: int
    iteration: int

    log_interval: int
    max_iterations: int

    shuffle: bool = field(jax_static=True)
    dataset: Any

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any

@dataclass(jax=True)
class StepState:
    epoch: int
    iteration: int

    log_interval: int
    max_iterations: int

    batch_dataset: Any
    batch_iterator: Any
    first_batch: Any

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any

@dataclass(jax=True)
class TrainResults:
    fn_params: Any
    fn_state: Any
    opt_state: Any

# A sentinel for not passing in state
# to the loss function
NO_STATE_TYPE=namedtuple('NoState',[])
NO_STATE=NO_STATE_TYPE()

@dataclass(jax=True)
class Trainer:
    loss_fn: Callable
    optimizer: optax.GradientTransformation = optax.adam(0.001)
    batch_size: int = field(default=32, jax_static=True)

    epochs : int = field(default=None, jax_static=True)
    max_iterations : int = field(default=None, jax_static=True)

    @jax.jit
    def _batch_loss_fn(self, fn_state, rng_key, batch, fn_params):
        if type(fn_state) == NO_STATE_TYPE:
            batch_loss_fn = jax.vmap(self.loss_fn,
                                     in_axes=(None, None, 0),
                                     axis_name='batch')
            loss, stats = batch_loss_fn(fn_params, rng_key, batch)
        else:
            batch_loss_fn = jax.vmap(self.loss_fn,
                                     in_axes=(None, None, None, 0),
                                     axis_name='batch')
            fn_state, loss, stats = batch_loss_fn(fn_params, fn_state,
                                                    rng_key, batch)
        loss = jnp.mean(loss)
        stats = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0), stats)
        return loss, (stats, fn_state)
    
    def _report(self, state, stats):
        logger.info("Training ({}/{}): {}", state.iteration + 1, state.max_iterations, stats)

    @jax.jit
    def _train_step(self, state): # first_batch is non-None
        rng_key, sk = jax.random.split(state.rng_key)

        if state.first_batch is None:
            batch = state.batch_dataset.get(state.batch_iterator)
            iterator = state.batch_dataset.next(state.batch_iterator)
        else: 
            batch = state.first_batch
            iterator = state.batch_iterator

        batch_fn = partial(self._batch_loss_fn, state.fn_state,
                        sk, batch)

        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.fn_params)

        fn_params = optax.apply_updates(state.fn_params, updates)

        if state.log_interval is not None:
            r = jnp.logical_or((state.iteration + 1) % state.log_interval == 0, state.iteration == 0)
            jax.lax.cond(r, self._report, lambda _0, _1: None, state, stats)

        return StepState(
            epoch=state.epoch,
            iteration=state.iteration + 1,
            max_iterations=state.max_iterations,
            log_interval=state.log_interval,

            first_batch=None,
            batch_dataset=state.batch_dataset,
            batch_iterator=iterator,

            rng_key=rng_key,
            fn_params=fn_params,
            fn_state=fn_state,
            opt_state=opt_state
        )
    
    @jax.jit
    def _train_epoch(self, state):
        if state.shuffle:
            rng_key, sk = jax.random.split(state.rng_key)
            dataset = state.dataset.shuffle(rng_key)
        else:
            rng_key = state.rng_key
            dataset = state.dataset

        first_batch, batch_dataset = dataset.batch(self.batch_size, ret_first=True)
        step_state = StepState(
            epoch=state.epoch, iteration=state.iteration,
            max_iterations=state.max_iterations,
            log_interval=state.log_interval,

            first_batch=first_batch,
            batch_dataset=batch_dataset,
            batch_iterator=batch_dataset.start,

            rng_key=rng_key,
            fn_params=state.fn_params,
            fn_state=state.fn_state,
            opt_state=state.opt_state
        )

        if first_batch is not None:
            step_state = self._train_step(step_state)

        if batch_dataset.length > 0:
            step_state = jax.lax.while_loop(lambda s: jnp.logical_and(s.iteration < s.max_iterations,
                                                jnp.logical_not(s.batch_dataset.is_end(s.batch_iterator))),
                                self._train_step, step_state)

        return EpochState(
            epoch=state.epoch + 1, iteration=step_state.iteration,
            max_iterations=state.max_iterations,
            log_interval=state.log_interval,

            shuffle=state.shuffle,
            dataset=state.dataset,
            rng_key=step_state.rng_key,
            fn_params=step_state.fn_params,
            fn_state=step_state.fn_state,
            opt_state=step_state.opt_state
        )
    
    @partial(jax.jit, static_argnums=(8,9))
    def train(self, dataset, rng_key,
                init_fn_params, init_fn_state=NO_STATE,
                init_opt_state=None,
                # can specify epochs either through
                # constructor or override in train() 
                # function
                epochs=None, max_iterations=None,
                shuffle=True, log_interval=None):
        
        # epochs and max_iterations can come from either
        # the trainer parameters or the train parameters
        epochs = epochs or self.epochs
        max_iterations = max_iterations or self.max_iterations

        if init_opt_state is None:
            init_opt_state = self.optimizer.init(init_fn_params)

        if not max_iterations and not epochs:
            raise ValueError("Must specify either number of epochs or iterations")
        
        rng_key, sub_key = jax.random.split(rng_key)
        if shuffle:
            dataset = dataset.shuffle(sub_key)

        num_batches = (dataset.length - 1) // self.batch_size + 1
        if max_iterations is None:
            max_iterations = num_batches*epochs
            if not jnp.isfinite(max_iterations):
                raise ValueError("Must train for a finite number of iterations")

        state = EpochState(
            epoch=0, iteration=0,
            max_iterations=max_iterations,
            log_interval=log_interval,
            
            shuffle=shuffle,
            dataset=dataset,

            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state
        )

        final_state = jax.lax.while_loop(
            lambda s: s.iteration < max_iterations,
            self._train_epoch, state
        )
        results = TrainResults(
            fn_params = final_state.fn_params,
            fn_state = final_state.fn_state,
            opt_state = final_state.opt_state,
        )
        return results