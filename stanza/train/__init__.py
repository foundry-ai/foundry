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
class TrainState:
    epoch: int
    iteration: int

    full_dataset: Any
    batch_dataset: Any
    first_batch: Any # None if not on first batch
    iterator: Any

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any


@dataclass(jax=True)
class TrainResults:
    fn_params: Any
    fn_state: Any
    opt_state: Any
    history: Any

# A sentinel for not passing in state
# to the loss function
NO_STATE_TYPE=namedtuple('NoState',[])
NO_STATE=NO_STATE_TYPE()

@dataclass(jax=True)
class Trainer:
    loss_fn: Callable
    optimizer: optax.GradientTransformation = optax.adam(0.001)
    batch_size: int = field(default=32, jax_static=True)
    preprocess_fn : Callable = None
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

    @jax.jit
    def _train_step(self, state): # first_batch is non-None
        rng_key, sk = jax.random.split(state.rng_key)

        if state.first_batch is None:
            batch = state.batch_dataset.get(state.iterator)
            iterator = state.batch_dataset.next(state.iterator)
        else: 
            batch = state.first_batch
            iterator = state.iterator

        batch_fn = partial(self._batch_loss_fn, state.fn_state,
                        sk, batch)

        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.fn_params)

        fn_params = optax.apply_updates(state.fn_params, updates)

        return TrainState(
            epoch=state.epoch,
            iteration=state.iteration + 1,

            first_batch=None,
            full_dataset=state.full_dataset,
            batch_dataset=state.batch_dataset,
            iterator=iterator,

            rng_key=rng_key,
            fn_params=fn_params,
            fn_state=fn_state,
            opt_state=opt_state
        ), stats
    
    @partial(jax.jit, static_argnums=(1,))
    def _adv_epoch(self, shuffle, state):
        rng_key, sk = jax.random.split(state.rng_key)
        if shuffle:
            full_dataset = state.full_dataset.shuffle(sk)
        else:
            full_dataset = state.full_dataset
        first_batch, dataset = full_dataset.batch(self.batch_size, ret_first=True)
        new_state = TrainState(
            epoch=state.epoch,
            iteration=state.iteration,

            full_dataset=state.full_dataset,
            batch_dataset=dataset,
            first_batch=first_batch,
            iterator=dataset.start,

            rng_key=rng_key,
            fn_params=state.fn_params,
            fn_state=state.fn_state,
            opt_state=state.opt_state
        )
        return self._train_step(new_state)

    @partial(jax.jit, static_argnums=(1,2))
    def _train_scan(self, pb, shuffle, state, _):
        new_epoch = state.batch_dataset.is_end(state.iterator)
        state, stats = jax.lax.cond(new_epoch,
            partial(self._adv_epoch, shuffle), self._train_step, state)

        # move the progress bar!
        if pb is not None:
            pb.inc(1, stats)
        return state, stats
    
    @partial(jax.jit, static_argnums=(8,9))
    def train(self, dataset, rng_key,
                init_fn_params, init_fn_state=NO_STATE,
                init_opt_state=None,
                # can specify epochs either through
                # constructor or override in train() 
                # function
                epochs=None, max_iterations=None,
                shuffle=True, show_pbar=True):
        
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

        first_batch, batch_dataset = dataset.batch(self.batch_size)
        if max_iterations is None:
            max_iterations = (batch_dataset.length + 1)*epochs
            if not jnp.isfinite(max_iterations):
                raise ValueError("Must train for a finite number of iterations")


        state = TrainState(
            epoch=0, iteration=0,

            full_dataset=dataset,
            batch_dataset=batch_dataset,
            first_batch=first_batch,
            iterator=batch_dataset.start,
            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state
        )
        if show_pbar:
            with pbar('trainer', total=max_iterations) as pb:
                # Do out the first step!
                state, stats = self._train_scan(pb, shuffle, state, None)
                final_state, stat_history = jax.lax.scan(partial(self._train_scan, pb), 
                                                         state, None, length=(max_iterations - 1))
        else:
            final_state, stat_history = jax.lax.scan(partial(self._train_scan, None), state, None, length=max_iterations)
            # for i in range(max_iterations):
            #     state, _ = self._train_scan(pb, state, None)
            # final_state = state
            # stat_history = None
        # return the final state and stat history
        logger.info("trainer", "Training complete")
        results = TrainResults(
            fn_params = final_state.fn_params,
            fn_state = final_state.fn_state,
            opt_state = final_state.opt_state,
            history = stat_history
        )
        return results