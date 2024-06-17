import flax.linen as nn
import jax.flatten_util

from . import activation
from functools import partial


class MLPClassifier(nn.Module):
    n_classes: int
    features: list[int] = (64, 64, 64)
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        x, _ = jax.flatten_util.ravel_pytree(x)
        h = getattr(activation, self.activation)
        for i, f in enumerate(self.features):
            x = nn.Dense(f)(x)
            x = h(x)
        x = nn.Dense(self.n_classes)(x)
        return x

MLPLarge = partial(MLPClassifier, features=[512, 512, 128])
MLPMedium = partial(MLPClassifier, features=[128, 128, 64])
MLPSmall = partial(MLPClassifier, features=[64, 32, 32])