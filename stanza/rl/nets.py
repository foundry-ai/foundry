import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import jax.numpy as jnp
import jax
import jax.scipy.stats.norm as norm


from typing import Any
from stanza.distribution import MultivariateNormalDiag

# TODO: typing for Gaussian Actor Critic


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



# BC losses for pre-training ac_net

def mean_loss(ac_net, state, action, loss_weight : jnp.array = None):
    pi, _ = ac_net(state)
    
    mean_flat, _ = jax.flatten_util.ravel_pytree(pi.mean)
    act_flat, _ = jax.flatten_util.ravel_pytree(action)
    if loss_weight is None:
        return jnp.linalg.norm(mean_flat - act_flat)
    else:
        return jnp.linalg.norm(jnp.multiply((mean_flat - act_flat),jnp.sqrt(loss_weight)))

_mvn_logpdf = jax.vmap(norm.logpdf)
#is this efficient
_mvn_inv_prec = jax.vmap(lambda x: 1/x)
_mvn_unit_reg = jax.vmap(lambda x: (x-1)**2)

def mle_loss(ac_net, state, action, barrier_reg = 0, to_unit_reg = 0):
    """ MLE loss for BC 
    Loss is given by - log likelihood + sum_i regularizer /sigma_i
    """
    pi, _ = ac_net(state)
    mean_flat, _ = jax.flatten_util.ravel_pytree(pi.mean)
    scale_diag_flat, _ = jax.flatten_util.ravel_pytree(pi.scale_diag)
    value_flat, _ = jax.flatten_util.ravel_pytree(action)
    log_pdf = _mvn_logpdf(value_flat, mean_flat, scale_diag_flat)
    #does this work?
    b_reg = barrier_reg * _mvn_inv_prec(scale_diag_flat)
    u_reg = to_unit_reg * _mvn_unit_reg(scale_diag_flat)
    return jnp.sum(b_reg + u_reg - log_pdf, -1)