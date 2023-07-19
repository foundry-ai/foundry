from stanza import Partial
from stanza.train import Trainer,  batch_loss, TrainState, TrainResults
from stanza.dataclasses import dataclass, field, replace, unpack
@dataclass(jax=True)
class BCState:
    ac_apply: callable
    train_state: TrainState

import jax 
import jax.numpy as jnp
from stanza.data.trajectory import Timestep


def l2_sample_loss(actor_apply, _state, a_params, _rng_key, sample : Timestep):
    #mean_flat, _ = jax.flatten_util.ravel_pytree(actor_apply(a_params,sample.observation))
    #action_flat, _ = jax.flatten_util.ravel_pytree(sample.action)
    #loss = jnp.square(mean_flat - action_flat).sum()
    #return _state, loss, {"loss": loss}
    return _state, 0., {"loss": 0.}



#data set must be of form
@dataclass(jax=True)
class BCTrainer:
    trainer: Trainer = field(
        default_factory=lambda: Trainer(batch_size=256)
    )
    
    def train(self, ac_apply, ac_params, dataset, rng_key, *,
              epochs=None, max_iterations=None, jit=True,
              init_opt_state=None, hooks=[]):
        
        l2_batch_loss = batch_loss(Partial(l2_sample_loss, ac_apply))

        result = self.trainer.train(
            l2_batch_loss, dataset,
            rng_key, ac_params,
            epochs=epochs, max_iterations=max_iterations, jit=jit,
            init_opt_state=init_opt_state,
            hooks=hooks
        )
        return result
    



