import jax
import jax.numpy as jnp


class NoiseInjector:

    def __init__(self, rng_key, sigma, base_policy):
        self.rng_key = rng_key
        self.sigma = sigma
        self.base_policy = base_policy
    
    def init_state(self, x0):
        return (self.rng_key, self.base_policy.init_state(x0))
    
    def __call__(self, x, policy_state):
        rng_key, base_policy_state = policy_state
        rng_key, sk = jax.random.split(rng_key)

        u_base, base_policy_state = self.base_policy(x, base_policy_state)

        # flatten u, add the noise, unflatten
        u_flat, unflatten = jax.flatten_util.ravel_pytree(u_base)
        noise = self.sigma * jax.random.normal(sk, u_flat.shape)
        u_flat = u_flat + noise
        u = unflatten(u_flat)

        return u, (rng_key, base_policy_state)


class SampleRandom:
    def __init__(self, rng_key, sample_fn):
        self.rng_key = rng_key
        self.sample_fn = sample_fn

    def init_state(self, x0):
        return self.rng_key
    
    def __call__(self, x, policy_state):
        rng_key = policy_state
        rng_key, sk = jax.random.split(rng_key)
        u = self.sample_fn(sk)
        return u, rng_key