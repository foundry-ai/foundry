import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import jax.numpy as jnp
import jax
import jax.scipy.stats.norm as norm


from typing import Any
from stanza.distribution import MultivariateNormalDiag

# TODO: typing for Gaussian Actor Critic
# TODO: real FLAX for call

class MLPActorCritic(nn.Module):
    action_sample: Any
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        action_flat, action_uf = jax.flatten_util.ravel_pytree(self.action_sample)
        x_flat, _ = jax.flatten_util.ravel_pytree(x)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x_flat)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            action_flat.shape[-1],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (action_flat.shape[0],))
        pi = MultivariateNormalDiag(action_uf(actor_mean), action_uf(jnp.exp(actor_logtstd)))

        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x_flat)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def transform_ac_to_mean(base_ac_apply):
    def new_apply(*args,**kwargs):
        return base_ac_apply(*args,**kwargs)[0].mean
    return new_apply

"""
_mvn_logpdf = jax.vmap(norm.logpdf)
#is this efficient
_mvn_inv_prec = jax.vmap(lambda x: 1/x)
_mvn_unit_reg = jax.vmap(lambda x: (x-1)**2)

def mle_loss(ac_net, state, action, barrier_reg = 0, to_unit_reg = 0):
    
    pi, _ = ac_net(state)
    v
    scale_diag_flat, _ = jax.flatten_util.ravel_pytree(pi.scale_diag)
    value_flat, _ = jax.flatten_util.ravel_pytree(action)
    log_pdf = _mvn_logpdf(value_flat, mean_flat, scale_diag_flat)
    #does this work?
    b_reg = barrier_reg * _mvn_inv_prec(scale_diag_flat)
    u_reg = to_unit_reg * _mvn_unit_reg(scale_diag_flat)
    return jnp.sum(b_reg + u_reg - log_pdf, -1)
    """