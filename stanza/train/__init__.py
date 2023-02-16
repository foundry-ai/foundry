from typing import List, Any
from stanza.util.logging import logger, pbar
from stanza import Partial
from functools import partial

import chex
import optax
import stanza

@dataclass
class TrainState:
    epoch: int
    iteration: int

    full_dataset: Any
    batch_dataset: Any
    iterator: Any

    rng_key: PRNGKey
    fn_params: Any
    fn_state: Any
    opt_state: Any


@dataclass
class TrainResults:
    fn_params: Any
    fn_state: Any
    opt_state: Any
    history: Any

# A sentinel for not passing in state
# to the loss function
NO_STATE_TYPE=namedtuple('NoState',[])
NO_STATE=NO_STATE_TYPE()

@dataclass(init=False)
class Trainer:
    loss_fn: Callable
    optimizer: optax.GradientTransformation = optax.adam(0.001)
    batch_size: int = 32
    preprocess_fn = None
    epochs = None
    max_iterations = None

    def __init__(self, loss_fn, optimizer=optax.adam(0.001),
                 batch_size=32, preprocess_fn=None,
                 epochs=None, max_iterations=None):
        # wrap to make sure loss_fn is jax-compatible
        self.loss_fn = stanza.fun(loss_fn)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.epochs = epochs
        self.max_iterations = max_iterations

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
    def _train_step(self, state):
        rng_key, sk = jax.random.split(state.rng_key)

        batch = state.batch_dataset.get(state.iterator)
        iterator = state.batch_dataset.next(state.iterator)

        batch_fn = partial(self._batch_loss_fn, state.fn_state,
                        sk, batch)

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
    
    @partial(jax.jit, static_argnums=(1,))
    def _adv_epoch(self, shuffle, state):
        rng_key, sk = jax.random.split(state.rng_key)
        if shuffle:
            full_dataset = state.full_dataset.shuffle(sk)
        dataset = full_dataset.batch(self.batch_size)
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

    @partial(jax.jit, static_argnums=(2,3))
    def _train_scan(self, pb, shuffle, state, _):
        new_epoch = state.batch_dataset.is_end(state.iterator)
        state = jax.lax.cond(new_epoch,
            partial(self._adv_epoch, shuffle), lambda x: x, state)
        state, stats = self._train_step(state)

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

        if self.preprocess:
            dataset = self.preprocess(dataset)

        if not max_iterations and not epochs:
            raise ValueError("Must specify either number of epochs or iterations")
        
        rng_key, sub_key = jax.random.split(rng_key)
        if shuffle:
            dataset = dataset.shuffle(sub_key)

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
        if show_pbar:
            with pbar('trainer', total=max_iterations) as pb:
                final_state, stat_history = jax.lax.scan(partial(self._train_scan, pb), state, None, length=max_iterations)
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