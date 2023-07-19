from stanza.distribution.common import Distribution
from stanza.dataclasses import dataclass, field, replace

import jax.numpy as jnp
import jax.scipy.stats.norm as norm
import jax

_mvn_logpdf = jax.vmap(norm.logpdf)

@dataclass(jax=True)
class MultivariateNormalDiag(Distribution):
    mean: jnp.ndarray
    scale_diag: jnp.ndarray

    def log_prob(self, value):
        mean_flat, _ = jax.flatten_util.ravel_pytree(self.mean)
        scale_diag_flat, _ = jax.flatten_util.ravel_pytree(self.scale_diag)
        value_flat, _ = jax.flatten_util.ravel_pytree(value)
        return jnp.sum(_mvn_logpdf(value_flat, mean_flat, scale_diag_flat), -1)

    def sample(self, rng_key):
        mean_flat, mean_uf = jax.flatten_util.ravel_pytree(self.mean)
        scale_diag_flat, _ = jax.flatten_util.ravel_pytree(self.scale_diag)
        sample = jax.random.normal(rng_key, mean_flat.shape) * scale_diag_flat + mean_flat
        return mean_uf(sample)
    
    def entropy(self):
        scale_diag_flat, _ = jax.flatten_util.ravel_pytree(self.scale_diag)
        a = jnp.sum(jnp.log(scale_diag_flat))/2
        b = scale_diag_flat.shape[-1]/2 * (1 + jnp.log(2*jnp.pi))
        return a + b
    
