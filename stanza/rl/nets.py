import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import jax.numpy as jnp
import jax

from typing import Any
from stanza.distribution import MultivariateNormalDiag


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
