import flax.linen as nn
import flax.linen.activation as activations

import jax
import jax.numpy as jnp

from typing import Sequence, Any

class MLP(nn.Module):
    features: Sequence[int]
    output_sample: Any
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = getattr(activations, self.activation)
        x, _ = jax.flatten_util.ravel_pytree(x)
        x = jnp.reshape(x, (-1,))
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
        out_flat, out_uf = jax.flatten_util.ravel_pytree(self.output_sample)
        x = nn.Dense(out_flat.shape[-1])(x)
        return out_uf(x)