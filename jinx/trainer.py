import jax
import tqdm
import jax.numpy as jnp
from jax.random import PRNGKey

import optax

from functools import partial
from typing import NamedTuple, List, Any
from collections import namedtuple
from dataclasses import dataclass

from jinx.logging import logger, pbar

class TrainState(NamedTuple):
    epoch: int
    iteration: int

    full_dataset: Any
    batch_dataset: Any
    iterator: Any

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any


# A sentinel for not passing in state
# to the loss function
NO_STATE_TYPE=namedtuple('NoState',[])
NO_STATE=NO_STATE_TYPE()

# Each term is fed in the original data and the output
# of "preprocess" of the original data item

class TrainResults(NamedTuple):
    fn_params: Any
    fn_state: Any
    opt_state: Any
    history: Any

class Trainer:
    # loss_fn takes arguments (params, state, rng_key, element)
    # and returns state, loss, stats
    # if state is not passed to the train function (using the NO_STATE sentinel),
    # it will not be passed to the loss function (or omitted)
    def __init__(self, loss_fn, optimizer=optax.adam(0.001), batch_size=32, 
                    preprocess=None, epochs=None, max_iterations=None):
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.loss_preprocess = None
        self.loss_fn = loss_fn

        self.preprocess = preprocess
        self.epochs = epochs
        self.max_iterations = max_iterations

    @partial(jax.jit, static_argnums=0)
    def _batch_loss_fn(self, fn_state, rng_key, batch, fn_params):
        if type(fn_state) == NO_STATE_TYPE:
            batch_loss_fn = jax.vmap(self.loss_fn, in_axes=(None, None, 0), axis_name='batch')
            loss, stats = batch_loss_fn(fn_params, rng_key, batch)
        else:
            batch_loss_fn = jax.vmap(self.loss_fn, in_axes=(None, None, None, 0), axis_name='batch')
            fn_state, loss, stats = batch_loss_fn(fn_params, fn_state, rng_key, batch)
        loss = jnp.mean(loss)
        stats = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0), stats)
        return loss, (stats, fn_state)

    @partial(jax.jit, static_argnums=0)
    def _train_step(self, state):
        rng_key, sk = jax.random.split(state.rng_key)

        batch = state.batch_dataset.get(state.iterator)
        iterator = state.batch_dataset.next(state.iterator)

        batch_fn = partial(self._batch_loss_fn, state.fn_state, sk, batch)

        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.fn_params)

        fn_params = optax.apply_updates(state.fn_params, updates)

        return TrainState(
            epoch=state.epoch,
            iteration=state.iteration + 1,

            full_dataset=state.full_dataset,
            batch_dataset=state.batch_dataset,
            iterator=iterator,

            rng_key=rng_key,
            fn_params=fn_params,
            fn_state=fn_state,
            opt_state=opt_state
        ), stats
    
    def _adv_epoch(self, state):
        rng_key, sk = jax.random.split(state.rng_key)
        dataset = state.full_dataset.shuffle(sk)
        dataset = dataset.batch(self.batch_size)
        return TrainState(
            epoch=state.epoch,
            iteration=state.iteration,

            full_dataset=state.full_dataset,
            batch_dataset=dataset,
            iterator=dataset.start,

            rng_key=rng_key,
            fn_params=state.fn_params,
            fn_state=state.fn_state,
            opt_state=state.opt_state
        )

    @partial(jax.jit,static_argnums=(0,1))
    def _train_scan(self, pb, state, _):
        # move the progress bar!

        new_epoch = state.batch_dataset.is_end(state.iterator)
        state = jax.lax.cond(new_epoch,
            self._adv_epoch, lambda x: x,  state)
        state, stats = self._train_step(state)
        pb.inc(1, stats)
        return state, stats
    
    # If epochs > 0, the dataset must be shuffleable
    def train(self, dataset, rng_key,
                init_fn_params, init_fn_state=NO_STATE,
                init_opt_state=None,
                epochs=None, max_iterations=None):
        
        # epochs and max_iterations can come from either
        # the trainer parameters or the train parameters
        epochs = epochs or self.epochs
        max_iterations = max_iterations or self.max_iterations

        if init_opt_state is None:
            init_opt_state = self.optimizer.init(init_fn_params)

        if self.preprocess:
            dataset = self.preprocess(dataset)

        if not max_iterations and not epochs:
            raise ValueError("Must specify either number of epochs or iterations")
        
        batch_dataset = dataset.batch(self.batch_size)
        if max_iterations is None:
            max_iterations = batch_dataset.length*epochs
            if not jnp.isfinite(max_iterations):
                raise ValueError("Must train for a finite number of iterations")


        state = TrainState(
            epoch=0, iteration=0,

            full_dataset=dataset,
            batch_dataset=batch_dataset,
            iterator=batch_dataset.start,
            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state
        )
        with pbar('trainer', total=max_iterations) as pb:
            final_state, stat_history = jax.lax.scan(partial(self._train_scan, pb), state, None, length=max_iterations)
        # return the final state and stat history
        logger.info("trainer", "Training complete")
        results = TrainResults(
            fn_params = final_state.fn_params,
            fn_state = final_state.fn_state,
            opt_state = final_state.opt_state,
            history = stat_history
        )
        return results