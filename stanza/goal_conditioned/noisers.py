import jax
from typing import Callable, Any
from jax.random import PRNGKey, split
from stanza.distribution.mvn import MultivariateNormalDiag
import jax.numpy as jnp 

Noiser = Callable[[PRNGKey, Any, int],Any]


def make_gaussian_noiser_scalar_variance(sigma):
    """
    @param sample : a function that takes a PRNGKey and returns a sample
    @param scale_diag : a scalar
    @return a noiser that takes a PRNGKey and a value and returns a noised value
    """
    #sample_flat, sample_uf = jax.flatten_util.ravel_pytree(sample)
    #zero_flat = jnp.zeros(sample.shape[0],)
    #var_flat = jnp.ones(sample.shape[0],) * sigma
    #mvn_dist = MultivariateNormalDiag(sample_uf(zero_flat), sample_uf(var_flat))
    def noiser(rng, value, timestep = 0):
        sigs = jax.tree_map(lambda x: sigma*jnp.ones_like(x), value)
        print(sigs)
        mvn = MultivariateNormalDiag(value, sigs)
        return mvn.sample(rng)

        noise = mvn_dist.sample(rng)
        noise_flat, _ = jax.flatten_util.ravel_pytree(noise)
        value_flat, _ = jax.flatten_util.ravel_pytree(value)
        return sample_uf(value_flat + noise_flat)
    return noiser




"""
creates_gaussian noiser of shape @sample
using a function to specialize the scedule 
"""

def make_gaussian_noiser_getter(var_fn : Callable[[int],float],sample):
    def noiser_getter(epoch_num: int):
        return make_gaussian_noiser_scalar_variance(sample, var_fn(epoch_num))
    return noiser_getter

NoiserGetter = Callable[[int],Noiser]
ProportionFunction2d = Callable[[int,int],float]
ProportionFunction1d = Callable[[int],float]


def even_proportions1d(i : int):
    return 1.

def even_proportions2d(i : int , j : int):
    return 1.

def choose_from_prop_fn(rng_key : PRNGKey, prop_function : ProportionFunction1d, num_choices : int):
    probs = jax.vmap(prop_function, in_axes=(0))(jnp.arange(num_choices))
    return jax.random.choice(rng_key, num_choices, p = probs)