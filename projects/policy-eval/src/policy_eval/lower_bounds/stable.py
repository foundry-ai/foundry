import foundry.numpy as npx
import foundry.random
import foundry.core as F

from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.core import Array
from foundry.core.dataclasses import dataclass
from foundry.util.registry import Registry

import flax.linen as nn

from typing import Sequence

def make_challenging_pair(mu=1/4):
    c_mu = 3/2*mu
    A_1 = npx.array([
        [1+mu, c_mu],
        [-c_mu, 1-2*mu]
    ])
    A_2 = npx.array([
        [-(1-mu/4), c_mu],
        [0, 1-2*mu]
    ])
    K_1 = npx.array([
        [-(1 + mu), -c_mu],
        [c_mu, 0]
    ])
    K_2 = npx.array([
        [(1 - mu/4), -c_mu],
        [0, 0]
    ])
    return (A_1, K_1), (A_2, K_2)


class MLPPredictor(nn.Module):
    d_output: int
    features: Sequence[int] = (64, 64, 64)

    def h(self,x):
        return x * jax.nn.sigmoid(x)
    
    @nn.compact
    def __call__(self, x):
        x, _ = jax.flatten_util.ravel_pytree(x)
        Dense = partial(nn.Dense, kernel_init = nn.initializers.lecun_normal(),
                         bias_init = nn.initializers.uniform())
        for i, f in enumerate(self.features):
            x = Dense(f)(x)
            x = self.h(x)
        x = Dense(self.d_output)(x)
        return x

class PerturbationModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x, _ = jax.flatten_util.ravel_pytree(x)
        Dense = partial(nn.Dense, kernel_init = nn.initializers.lecun_normal(),
                         bias_init = nn.initializers.uniform())
        for i, f in enumerate(self.features):
            x = Dense(f)(x)
            x = self.h(x)
        x = Dense(1)(x)
        return x

@F.jit
def default_bump(x):
    x_norm = jnp.linalg.norm(x)
    return jax.lax.cond(x_norm < 1, 
        lambda: 1.,
        lambda: jax.lax.cond(x_norm > 2, 
        lambda: 0.,
        lambda: jnp.exp(- 1 / (1 - (x - 1)^2))
    ))

@dataclass
class EmbedEnvironment:
    A: Array
    tau: Array

    vars: Any
    bump: callable = default_bump
    delta: float = 0.5
    omega: float = 1
    d: int = 16

    @staticmethod
    def create_model():
        return PerturbationModel([16, 16, 16])

    def sample_state(self, rng_key):
        return npx.zeros((2 + d,))

    def sample_action(self, rng_key):
        return npx.zeros((2 + d,))

    def reset(rng_key):
        Z_rng, z_sphere_rng, z_d_rng, w_rng = foundry.random.split(rng_key, 4)
        Z = foundry.random.bernoulli(Z_rng)

        z = foundry.random.normal(z_sphere_rng, (self.d,))
        z = z / npx.linalg.norm(z)
        z = z * foundry.random.uniform(z_d_rng)

        z = npx.concatenate((npx.array([3, 0]), z), axis=0)

        w = foundry.random.normal(w_rng, (self.d,))
        w = w / npx.linalg.norm(w)

        return (1 - Z) * z + Z * w

    def step(self, state, input, rng_key):
        model = self.create_model()
        g_out = model.apply(self.vars, state[2:])
        restrict = self.bump(state - jnp.zeros_like(state).at[2].set(1))
        state_lower = self.A @ state[:2]
        perturbation = (
            - tau * restrict * g_out
            + self.omega * tau**2 * restrict * (g_out - input[0]*self.bump(input)/tau)
        )
        state_lower = state_lower.at[0].add(perturbation)
        state = npx.concatenate((state_lower, npx.zeros((d,))), axis=0)
        return state + input

@dataclass
class EmbedExpert(Policy):
    K: Array
    tau: Array
    vars: Any
    bump: callable = default_bump
    d: int = 16

    def __call__(self, input : PolicyInput) -> PolicyOutput:
        obs = input.observation
        input_upper = obs
        input = npx.concatenate((input_upper, input_lower), axis=0)
        return PolicyOutput(input)

def create_environment(rng_key, d, pair_first):
    M1, M2 = make_challenging_pair()
    A, K = M1 if pair_first else M2
    model = EmbedEnvironment(A, K, d)
    vars = model.init(rng_key, npx.zeros((d,)))
    return EmbedEnvironment(A, K, d, vars)

def register_all(registry : Registry, prefix=None):
    registry.register(
        "lower_bound/stable/1", 
        partial(create_environment, rng_key=foundry.random.key(42), d=16, pair_first=True),
        prefix=prefix
    )
    registry.register(
        "lower_bound/stable/2", 
        partial(create_environment, rng_key=foundry.random.key(42), d=16, pair_first=False),
        prefix=prefix
    )