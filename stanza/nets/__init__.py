from typing import Callable
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
