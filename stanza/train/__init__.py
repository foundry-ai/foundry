from stanza import Partial
from stanza.util.dataclasses import dataclass, field, replace
from stanza.data import PyTreeData

from jax.random import PRNGKey

from typing import Callable, Any
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

    first_batch: Any
    batch_iterator: Any

    last_stats : Any

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
    optimizer: optax.GradientTransformation = optax.adam(1e-3)

    batch_size: int = field(default=32, jax_static=True)
    # optional, if not set
    # must be passed into train()
    epochs : int = field(default=None, jax_static=True)
    max_iterations : int = field(default=None, jax_static=True)

    @jax.jit
    def _batch_loss_fn(self, loss_fn, fn_state, 
                    rng_key, batch, fn_params):
        if type(fn_state) == NO_STATE_TYPE:
            batch_loss_fn = jax.vmap(loss_fn,
                                     in_axes=(None, None, 0),
                                     axis_name='batch')
            loss, stats = batch_loss_fn(fn_params, rng_key, batch)
        else:
            batch_loss_fn = jax.vmap(loss_fn,
                                     in_axes=(None, None, None, 0),
                                     axis_name='batch')
            fn_state, loss, stats = batch_loss_fn(fn_params, fn_state,
                                                    rng_key, batch)
        loss = jnp.mean(loss)
        stats = jax.tree_util.tree_map(lambda x: jnp.mean(x,axis=0), stats)
        return loss, (stats, fn_state)

    def _train_step(self, state, *, loss_fn, batch_dataset, hooks):
        rng_key, sk = jax.random.split(state.rng_key)

        if state.first_batch is None:
            batch_data = batch_dataset.get(state.batch_iterator)
            batch_iterator = batch_dataset.next(state.batch_iterator)
        else: 
            batch_data = state.first_batch
            batch_iterator = batch_dataset.start

        batch = PyTreeData.from_data(batch_data)
        batch_fn = Partial(self._batch_loss_fn, loss_fn,
                            state.fn_state,
                            sk, batch.data)

        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.fn_params)

        fn_params = optax.apply_updates(state.fn_params, updates)

        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            total_iteration=state.total_iteration + 1,
            first_batch=None,
            batch_iterator=batch_iterator,
            last_stats=stats,
            rng_key=rng_key,
            fn_params=fn_params, fn_state=fn_state, opt_state=opt_state)

        for h in hooks:
            state = h(state)
        return state

    @partial(jax.jit, static_argnames=("shuffle",))
    def _train_epoch(self, state, *, 
                loss_fn, dataset, hooks, shuffle):
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)

        dataset = dataset if not shuffle else dataset.shuffle(sk)
        first_batch, batch_dataset = dataset.batch(self.batch_size)

        first_step_state = replace(state,
            first_batch=first_batch,
            batch_iterator=None)
        # handle the first batch
        # separately
        cond_fn = lambda s: jnp.logical_and(s.total_iteration < s.max_iteration,
                                jnp.logical_not(batch_dataset.is_end(s.batch_iterator)))
        step_fn = Partial(self._train_step,
                    loss_fn=loss_fn,
                    batch_dataset=batch_dataset,
                    hooks=hooks)

        step_state = step_fn(first_step_state)
        if batch_dataset.length > 0:
            step_state = jax.lax.while_loop(cond_fn, step_fn, step_state)

        return replace(state,
            epoch=state.epoch+1,
            epoch_iteration=0,
            last_stats=step_state.last_stats,
            total_iteration=step_state.total_iteration,
            rng_key=step_state.rng_key,
            fn_params=step_state.fn_params,
            fn_state=step_state.fn_state,
            opt_state=step_state.opt_state
        )
    
    @stanza.jit(static_argnames=("epochs", "max_iterations", "shuffle"))
    def train(self, loss_fn, dataset, rng_key,
                init_fn_params, init_fn_state=NO_STATE,
                init_opt_state=None,
                *,
                # can specify epochs either through
                # constructor or override in train() 
                # function
                epochs=None, max_iterations=None,
                shuffle=True, hooks=[]):
        
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

        if max_iterations is None:
            num_batches = (dataset.length - 1) // self.batch_size + 1
            max_iterations = num_batches*epochs

        state = TrainState(
            epoch=0, 
            max_epoch=epochs,
            epoch_iteration=0,
            total_iteration=0,
            max_iteration=max_iterations,

            # Used at the step level
            # not the state level
            first_batch=None,
            batch_iterator=None,

            last_stats=None,

            rng_key=rng_key, 
            fn_params=init_fn_params,
            fn_state=init_fn_state,
            opt_state=init_opt_state
        )

        epoch_fn = Partial(
            self._train_epoch,
            loss_fn=Partial(loss_fn),
            dataset=dataset,
            hooks=hooks,
            shuffle=shuffle
        )

        # do the first epoch
        # by hand (a) to handle
        # last_stats=None and (b)
        # to make debugging
        # easier
        state = epoch_fn(state)

        final_state = jax.lax.while_loop(
            lambda s: s.total_iteration < max_iterations,
            epoch_fn, state
        )

        for h in hooks:
            final_state = h(final_state)

        results = TrainResults(
            fn_params = final_state.fn_params,
            fn_state = final_state.fn_state,
            opt_state = final_state.opt_state,
        )
        return results