from typing import Any
from stanza.distribution.mvn import MultivariateNormalDiag
import jax.numpy as jnp 
from stanza.dataclasses import dataclass, field, replace
from stanza.distribution.common import Distribution
import jax

class ScalingDistribution(Distribution):
    def first_moment():
        pass
    def sample(rnk_key):
        pass

@dataclass(jax=True)
class UniformScaling(ScalingDistribution):
    min_val : float = 0.0
    max_val : float = 1.0
    def first_moment(self):
        return (self.min_val + self.max_val)/2
    def sample(self,rnk_key):
        return jax.random.uniform(rnk_key,shape=(),minval=self.min_val,maxval=self.max_val)
    
@dataclass(jax=True)
class GaussianScaling(ScalingDistribution):
    mean : float = 0.0
    std : float = 1.0
    def first_moment(self):
        return self.mean
    def sample(self,rnk_key):
        return jax.random.normal(rnk_key,shape=(),mean=self.mean,std=self.std)

@dataclass(jax=True)
class ScaledMVNDiag(Distribution):
    g_mean: jnp.ndarray
    g_scale_diag: jnp.ndarray
    scalar_dist: Distribution
    should_rescale: bool = field(default=False,jax_static=True)

    def sample(self,rnk_key):
        g_sample = MultivariateNormalDiag(mean = self.g_mean,scale_diag = self.g_scale_diag).sample(rnk_key)
        scale_sample = self.scalar_dist.sample(rnk_key)
        sample = g_sample * scale_sample
        return sample if not self.should_rescale else (sample / self.scalar_dist.first_moment())

