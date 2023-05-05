from stanza import Partial
from stanza.util.dataclasses import dataclass, field, replace
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

# A sentinel for not passing in state
# to the loss function
NO_STATE_TYPE=namedtuple('NoState',[])
NO_STATE=NO_STATE_TYPE()

@dataclass(jax=True, kw_only=True)
class Trainer:
    optimizer: optax.GradientTransformation = optax.adam(1e-3)

    batch_size: int = field(default=32, jax_static=True)
    # optional, if not set
    # must be passed into train()
    epochs : int = field(default=None)
    max_iterations : int = field(default=None)

    # TODO: This batched loss function
    # does not handle the state properly
    # since masked-out entries can influence
    # the state
    @jax.jit
    def _batch_loss_fn(self, loss_fn, fn_state, 
                    rng_key, batch, batch_n, fn_params):
        logger.trace("Tracing batch loss", only_tracing=True)
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

        sample_mask = jnp.arange(loss.shape[0]) < batch_n
        def mean(x):
            x = jnp.where(sample_mask, x, 0)
            return jnp.sum(x, axis=0)/batch_n
        loss = mean(loss)
        stats = jax.tree_util.tree_map(mean, stats)
        return loss, (stats, fn_state)
    
    def _run_hooks(self, state, hooks):
        new_hook_states = []
        for h, hs in zip(hooks, state.hook_states):
            hs, state = h(hs, state)
            new_hook_states.append(hs)
        state = replace(state, hook_states=new_hook_states)
        return state

    @jax.jit
    def _train_step(self, state, *, loss_fn, batch_dataset, hooks):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch_data = batch_dataset.get(state.batch_iterator)
        batch_iterator = batch_dataset.next(state.batch_iterator)

        # Specify a buffer size!
        batch = PyTreeData.from_data(batch_data,
                            buffer_size=self.batch_size)

        batch_fn = Partial(self._batch_loss_fn, loss_fn,
                            state.fn_state,
                            sk, batch.data, batch.n)

        grads, (stats, fn_state) = jax.grad(batch_fn, has_aux=True)(state.fn_params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.fn_params)

        fn_params = optax.apply_updates(state.fn_params, updates)

        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            total_iteration=state.total_iteration + 1,
            batch_iterator=batch_iterator,
            last_stats=stats,
            rng_key=rng_key,
            fn_params=fn_params, fn_state=fn_state, opt_state=opt_state)
        state = self._run_hooks(state, hooks)
        # Make none after the step
        # so we keep the same state shape in + out
        state = replace(state, last_stats=None)
        return state

    @partial(jax.jit, static_argnames=("shuffle",))
    def _train_epoch(self, state, *, 
                loss_fn, dataset, hooks, shuffle):
        logger.trace("Tracing epoch step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)

        dataset = dataset if not shuffle else dataset.shuffle(sk)
        batch_dataset = dataset.batch(self.batch_size)

        first_step_state = replace(state, batch_iterator=batch_dataset.start)
        cond_fn = lambda s: jnp.logical_and(s.total_iteration < s.max_iteration,
                                jnp.logical_not(batch_dataset.is_end(s.batch_iterator)))
        step_fn = Partial(self._train_step,
                    loss_fn=loss_fn,
                    batch_dataset=batch_dataset,
                    hooks=hooks)

        # call the hooks before each epoch
        step_state = self._run_hooks(first_step_state, hooks)
        # run first step separately
        step_state = step_fn(step_state)
        step_state = jax.lax.while_loop(cond_fn, step_fn, step_state)

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
    
    @staticmethod
    def total_iterations(dataset, batch_size, epochs):
        num_batches = (dataset.length - 1) // batch_size + 1
        return num_batches * epochs
    
    @stanza.jit(static_argnames=("shuffle",))
    def train(self, loss_fn, dataset, rng_key,
                init_fn_params, init_fn_state=NO_STATE,
                init_opt_state=None,
                *,
                # can specify epochs either through
                # constructor or override in train() 
                # function
                epochs=None, max_iterations=None,
                shuffle=True, hooks=[]):
        logger.trace("Tracing training", only_tracing=True)
        logger.trace("Starting training")
        
        # epochs and max_iterations can come from either
        # the trainer parameters or the train parameters
        epochs = self.epochs if epochs is None else epochs
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        if init_opt_state is None:
            init_opt_state = self.optimizer.init(init_fn_params)
        if max_iterations is None and epochs is None:
            raise ValueError("Must specify either number of epochs or iterations")
        
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
            batch_iterator=None,

            last_stats=None,
            hook_states=[None]*len(hooks),

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
        # the first hook states and (b)
        # to make debugging easier
        state = epoch_fn(state)
        final_state = jax.lax.while_loop(
            lambda s: s.total_iteration < max_iterations,
            epoch_fn, state
        )
        final_state = self._run_hooks(final_state, hooks)
        logger.trace("Done tracing training", only_tracing=True)
        results = TrainResults(
            fn_params = final_state.fn_params,
            fn_state = final_state.fn_state,
            opt_state = final_state.opt_state,
            hook_states = final_state.hook_states
        )
        return results