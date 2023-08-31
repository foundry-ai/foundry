import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence


class MLP(nn.Module):
  features: Sequence[int]
  activation: str = "relu"

  @nn.compact
  def __call__(self, x):
    activation = getattr(nn, self.activation)
    x = jnp.atleast_1d(x)
    x = jnp.reshape(x, (-1,))
    for feat in self.features[:-1]:
      x = activation(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x