import flax.linen as nn
import jax.flatten_util

from . import activation
from functools import partial
from foundry.util.registry import Registry


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

MLPLargeClassifier = partial(MLPClassifier, features=[512, 512, 128])
MLPMediumClassifier = partial(MLPClassifier, features=[128, 128, 64])
MLPSmallClassifier = partial(MLPClassifier, features=[64, 32, 32])

models = Registry()
models.register("LargeClassifier", MLPLargeClassifier)
models.register("MediumClassifier", MLPMediumClassifier)
models.register("SmallClassifier", MLPSmallClassifier)