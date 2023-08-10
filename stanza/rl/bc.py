from stanza import Partial
from stanza.train import Trainer,  batch_loss, TrainState, TrainResults
from stanza.dataclasses import dataclass, field, replace, unpack
import optax
from jax.random import PRNGKey


import jax 
import jax.numpy as jnp
from stanza.data.trajectory import Timestep
from typing import Any, Callable

@dataclass(jax=True)
class BCState:
    ac_apply: callable
    train_state: TrainState

ActorApply = Any
State = Any
ActorParams = Any
GoalReward = Callable[[ActorApply,State,ActorParams,],float]

#note, these are non-vmapped
class BCLoss:
    def loss(actor_apply, _state, a_params, _rng_key : PRNGKey, 
             sample : Timestep):
        raise NotImplementedError("Must impelement reset()")
    

@dataclass(jax=True)
class L2BCLoss(BCLoss):
    #TODO: allow for reweighting the loss in diff ways
    def loss(actor_apply, _state, a_params, _rng_key : PRNGKey, sample : Timestep):
        mean_flat, _ = jax.flatten_util.ravel_pytree(actor_apply(a_params,sample.observation).mean)
        action_flat, _ = jax.flatten_util.ravel_pytree(sample.action)
        loss = jnp.square(mean_flat - action_flat).sum()
        return _state, loss, {"loss": loss}



@dataclass(jax=True)
class MleBCLoss(BCLoss):
    #TODO: allow for reweighting the loss in diff ways
    def loss(actor_apply, _state, a_params, _rng_key : PRNGKey, sample : Timestep):
        loss = -1 * actor_apply(a_params, sample.observation).log_prob(sample.action)
        return _state, loss, {"loss": loss}
    
def l2_loss(actor_apply, _state, a_params, _rng_key : PRNGKey, sample : Timestep):
        mean_flat, _ = jax.flatten_util.ravel_pytree(actor_apply(a_params,sample.observation).mean)
        action_flat, _ = jax.flatten_util.ravel_pytree(sample.action)
        loss = jnp.square(mean_flat - action_flat).sum()
        return _state, loss, {"loss": loss}

#data set must be of form
@dataclass(jax=True)
class BCTrainer:
    loss_fn : BCLoss = L2BCLoss()
    trainer: Trainer = field(
        default_factory=lambda: Trainer(
        optimizer = optax.adam(optax.cosine_decay_schedule(1e-3, 10000, 1e-5))
        )
    )
    def train(self, ac_apply, ac_params, dataset, rng_key, *,
              epochs=None, max_iterations=None, jit=True,
              init_opt_state=   None, hooks=[]):
        
        batch_loss_fn = batch_loss(Partial(self.loss_fn.__class__.loss, ac_apply))
        #recall that a batch_loss takes in (fn_fn_state, fn_params, rng_key, batch)
        result = self.trainer.train(
            batch_loss_fn, dataset,
            rng_key, ac_params,
            epochs=epochs, max_iterations=max_iterations, jit=jit,
            init_opt_state=init_opt_state,
            hooks=hooks
        )
        return result
    



