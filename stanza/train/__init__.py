from stanza import Partial
from stanza.dataclasses import dataclass, field, \
    replace, unpack, combine
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

@dataclass(jax=True, kw_only=True)
class TrainConfig:
    loss_fn: Callable = None
    rng_key: PRNGKey = None

    init_params: Any = None
    init_state: Any = None
    init_opt_state: Any = None

    optimizer: optax.GradientTransformation = optax.adam(1e-3)
    batch_size: int = field(default=32, jax_static=True)
    # optional, if not set
    # must be passed into train()
    max_epochs : int = field(default=None)
    max_iterations : int = field(default=None)
    train_hooks: List[Callable] = field(default_factory=list)

@dataclass(jax=True, kw_only=True)
class TrainState(LoopState):
    config: TrainConfig

    epoch: int
    max_epochs: int
    epoch_iteration: int

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


def _trainer_loss_fn(loss_fn, fn_state, rng_key, batch, fn_params):
    fn_state, loss, stats = loss_fn(fn_state, fn_params, rng_key, batch)
    return loss, (fn_state, stats)

@dataclass(jax=True, kw_only=True)
class Trainer(TrainConfig):
    @staticmethod
    def train_step(state, batch):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch = PyTreeData.from_data(batch,
                    buffer_size=state.config.batch_size)
        batch_fn = Partial(_trainer_loss_fn, 
                state.config.loss_fn, state.fn_state, 
                sk, batch.data)
        batch_grad_fn = jax.grad(batch_fn, has_aux=True)
        grads, (fn_state, stats) = batch_grad_fn(state.fn_params)

        if fn_state is not None or state.fn_state is not None:
            chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)

        updates, opt_state = state.config.optimizer.update(grads, 
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
    
    @classmethod
    def train_step_with_hooks(cls, state, batch):
        state = cls.train_step(state, batch)
        state = run_hooks(state)
        return state

    @classmethod
    def train_epoch(cls, state, dataset, *, jit=True):
        logger.trace("Tracing epoch step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)

        shuffled_dataset = dataset.shuffle(sk)
        batch_dataset = shuffled_dataset.batch(state.config.batch_size)

        step_fn = cls.train_step_with_hooks
        state = run_hooks(state)
        state = batch_dataset.scan(step_fn, state, jit=jit,
                limit=state.max_iterations - state.iteration)
        return replace(state,
            epoch=state.epoch+1,
            epoch_iteration=0,
        )

    @classmethod
    def run(cls, state, dataset, jit=True):
        logger.trace("Tracing training", only_tracing=True)
        if jit:
            train_epoch = jax.jit(cls.train_epoch)
            epoch_fn = Partial(train_epoch, dataset=dataset)
        else:
            epoch_fn = Partial(cls.train_epoch, dataset=dataset, jit=False)
        # populate the last_stats if necessary
        state = loop(epoch_fn, state, jit=jit)
        state = run_hooks(state)
        logger.trace("Done tracing training", only_tracing=True)
        return state
    
    def init(self, data_sample, config=None, init_hooks=True, **kwargs):
        if config is None:
            config = combine(TrainConfig, self, kwargs)
        else:
            config = replace(config, **kwargs)
        if config.max_iterations is None:
            raise RuntimeError("max_iterations must be set")
        if config.init_opt_state is None:
            config = replace(config, 
                init_opt_state=config.optimizer.init(config.init_params))
        batch_sample = jax.tree_map(
            lambda x: jnp.repeat(jnp.array(x)[jnp.newaxis, ...], self.batch_size, axis=0),
            data_sample
        )
        _, _, stats = config.loss_fn(config.init_state, config.init_params, 
                            PRNGKey(42), batch_sample)
        stats = jax.tree_map(jnp.zeros_like, stats)

        state = TrainState(
            iteration=0,
            max_iterations=config.max_iterations,
            hooks=config.train_hooks,
            hook_states=None,
            epoch=0, 
            max_epochs=config.max_epochs,
            epoch_iteration=0,

            config=config,

            last_stats={"train":stats},

            rng_key=config.rng_key, 
            fn_params=config.init_params,
            fn_state=config.init_state,
            opt_state=config.init_opt_state
        )
        if init_hooks:
            state = _init_hooks(state)
        return state
    
    def train(self, dataset, loss_fn=None, *, jit=True, **kwargs):
        if loss_fn is not None:
            kwargs["loss_fn"] = loss_fn
        config = combine(TrainConfig, self, kwargs)
        if config.max_iterations is None and config.max_epochs is not None:
            iter_per_epoch = len(dataset) // config.batch_size
            max_iterations = iter_per_epoch * config.max_epochs
            kwargs["max_iterations"] = max_iterations
        state = self.init(dataset[0], **kwargs)
        run = jax.jit(self.run) if jit else partial(self.run, jit=False)
        state = run(state, dataset)
        results = TrainResults(
            fn_params = state.fn_params,
            fn_state = state.fn_state,
            opt_state = state.opt_state,
            hook_states = state.hook_states
        )
        return results

@dataclass(jax=True)
class SAMConfig(TrainConfig):
    sub_optimizer: optax.GradientTransformation = optax.sgd(1e-5)
    init_sub_opt_state: Any = None
    normalize: bool = field(default=True, jax_static=True)
    # Number of iterations to run SAM for
    sam_iterations: int = field(default=None)
    resample: bool = field(default=False, jax_static=True)

@dataclass(jax=True)
class SAMTrainState(TrainState):
    sub_opt_state: Any

@dataclass(jax=True)
class SAMTrainResults(TrainResults):
    sub_opt_state: Any


@dataclass(jax=True)
class SAMTrainer(Trainer):
    sub_optimizer: optax.GradientTransformation = optax.scale(1e-2)
    init_sub_opt_state: Any = None
    normalize: bool = field(default=True, jax_static=True)
    # Number of iterations to run SAM for
    sam_iterations: int = field(default=None)
    resample: bool = field(default=False, jax_static=True)

    @staticmethod
    def train_step(state, batch):
        logger.trace("Tracing train step", only_tracing=True)
        rng_key, sk = jax.random.split(state.rng_key)
        batch = PyTreeData.from_data(batch,
                    buffer_size=state.config.batch_size)
        batch_fn = Partial(_trainer_loss_fn, 
                state.config.loss_fn, state.fn_state, 
                sk, batch.data)
        batch_grad_fn = jax.grad(batch_fn, has_aux=True)
        sam_batch_grad_fn = batch_grad_fn

        grads, _ = sam_batch_grad_fn(state.fn_params)
        if state.config.normalize:
            global_norm = optax.global_norm(grads)
            grads = jax.tree_map(lambda x: x / global_norm, grads)
        updates, sub_opt_state = state.config.sub_optimizer.update(grads, 
                            state.sub_opt_state, state.fn_params)
        sub_fn_params = optax.apply_updates(state.fn_params, updates)
        grads, (fn_state, stats) = batch_grad_fn(sub_fn_params)
        chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)
        updates, opt_state = state.config.optimizer.update(grads, 
                            state.opt_state, state.fn_params)
        fn_params = optax.apply_updates(state.fn_params, updates)

        if fn_state is not None or state.fn_state is not None:
            chex.assert_trees_all_equal_shapes_and_dtypes(fn_state, state.fn_state)

        state = replace(state,
            epoch_iteration=state.epoch_iteration + 1,
            iteration=state.iteration+ 1,
            last_stats=replace(
                state.last_stats,
                train=stats
            ), rng_key=rng_key,
            fn_params=fn_params, 
            fn_state=fn_state, opt_state=opt_state,
            sub_opt_state=sub_opt_state)
        return state

    def init(self, data_sample, config=None, init_hooks=True, 
                    **kwargs):
        if config is None:
            config = combine(SAMConfig, self, kwargs)
        else:
            config = replace(config, **kwargs)
        state = super().init(data_sample, config=config,
                            init_hooks=init_hooks)
        if config.init_sub_opt_state is None:
            init_sub_opt_state = config.sub_optimizer.init(state.fn_params)
        state = SAMTrainState(**unpack(state),
                      sub_opt_state=init_sub_opt_state)
        return state

# utility for batch loss

@jax.jit
def _batch_loss_fn(loss_fn, fn_state, fn_params,
                        rng_key, batch):
    logger.trace("Tracing batch loss", only_tracing=True)
    batch_loss_fn = jax.vmap(loss_fn,
                    in_axes=(None, None, 0, 0),
                    out_axes=(None, 0, 0),
                    axis_name='batch')
    n = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
    rng_key = jax.random.split(rng_key, n)
    fn_state, loss, stats = batch_loss_fn(fn_state, fn_params,
                                            rng_key, batch)
    loss = jnp.mean(loss)
    stats = jax.tree_map(jnp.mean, stats)
    return fn_state, loss, stats

def batch_loss(loss_fn):
    return Partial(_batch_loss_fn, Partial(loss_fn))