import foundry.numpy as npx
import foundry.random
import foundry.core as F
import foundry.policy

from foundry.core import tree
from foundry.graphics import canvas
from foundry.env.core import Environment, ImageRender
from foundry.data.sequence import SequenceData, Step
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.core import Array
from foundry.datasets.env import EnvDataset
        
from foundry.core.dataclasses import dataclass
from foundry.util.registry import Registry

import flax.linen as nn
import jax

from typing import Any, Sequence
from functools import partial

def make_challenging_pair(mu=1/8):
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

class PerturbationModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x, _ = jax.flatten_util.ravel_pytree(x)
        Dense = partial(nn.Dense, kernel_init = nn.initializers.lecun_normal(),
                         bias_init = nn.initializers.uniform())
        h = lambda x: jax.nn.tanh(x)
        for i, f in enumerate(self.features):
            x = Dense(f)(x)
            x = h(x)
        x = Dense(1)(x)
        return x.squeeze()/200

@F.jit
def default_bump(x):
    x_norm = npx.linalg.norm(x)
    return jax.lax.cond(x_norm < 1, 
        lambda: 1.,
        lambda: jax.lax.cond(x_norm > 2, 
        lambda: 0.,
        lambda: npx.exp(- 1 / (1 - (x_norm - 1)**2))
    ))

@dataclass
class EmbedEnvironment(Environment):
    A: Array
    K: Array

    vars: Any
    d: int

    tau: Array = 0.001
    bump: callable = default_bump
    delta: float = 0.3
    omega: float = 1

    @staticmethod
    def create_model():
        return PerturbationModel([16, 16])

    def sample_state(self, rng_key):
        return npx.zeros((2 + self.d,))

    def sample_action(self, rng_key):
        return npx.zeros((2 + self.d,))
    
    def reset(self, rng_key):
        Z_rng, z_sphere_rng, z_d_rng, w_rng, w_d_rng = foundry.random.split(rng_key, 5)
        Z = foundry.random.bernoulli(Z_rng)

        # generate uniform over sphere z
        z = foundry.random.normal(z_sphere_rng, (self.d,))
        z = z / npx.linalg.norm(z)
        z = z * foundry.random.uniform(z_d_rng)

        z = npx.concatenate((npx.zeros((2,)), z), axis=0)
        z = z.at[2].add(3)

        w = foundry.random.normal(w_rng, (self.d + 1,))
        w = w / npx.linalg.norm(w)
        w = w * foundry.random.uniform(w_d_rng)

        w = npx.concatenate((npx.zeros((1,)), w))
        return (1 - Z) * z + Z * w

    def step(self, state, input, rng_key):
        model = self.create_model()
        g_out = model.apply(self.vars, state[2:])
        restrict = self.bump(state - npx.zeros_like(state).at[2].set(3))
        state_lower = self.A @ state[:2]
        perturbation = (
            - self.tau * restrict * g_out
            + self.omega * self.tau**2 * restrict * g_out - self.tau * input[0]*self.bump(input)
        )
        state_lower = state_lower.at[0].add(perturbation)
        state = npx.concatenate((state_lower, npx.zeros((self.d,))), axis=0)
        next_state = state.at[:2].add(input[:2])

        # To prevent NaN explosion...
        next_state = npx.clip(next_state, -1_000_000_000, 1_000_000_000)
        return next_state
    
    def observe(self, state, config=None):
        return state
    
    def reward(self, state, action, next_state):
        return -npx.abs(next_state[0])

    def combined_reward(self, states, actions):
        pre_states, actions, post_states = (
            tree.map(lambda x: x[:-1], states),
            tree.map(lambda x: x[:-1], actions),
            tree.map(lambda x: x[1:], states)
        )
        rewards = F.vmap(self.reward)(
            pre_states, actions, post_states
        )
        return npx.min(rewards)

    @F.jit
    def render(self, state, config):
        if isinstance(config, ImageRender):
            image = npx.ones((config.height, config.width, 3))
            loc = (state[:2]/6 + 0.5)*npx.array([config.width, config.height])
            image = canvas.paint(
                image,
                canvas.fill(canvas.segment((0, config.height/2), (config.width, config.height/2))),
                canvas.fill(canvas.segment((config.width/2, 0), (config.width/2, config.height))),
                canvas.fill(canvas.circle(loc, 3), color=canvas.colors.Blue),
            )
            return image

@dataclass
class EmbedExpert(Policy):
    K: Array
    vars: Any
    tau: Array
    bump: callable = default_bump

    def __call__(self, input : PolicyInput) -> PolicyOutput:
        obs = input.observation
        input = self.K @ obs[:2] # the data
        input = npx.concatenate((input, npx.zeros((obs.shape[0] - 2,))), axis=0)

        model = EmbedEnvironment.create_model()
        g_out = model.apply(self.vars, obs[2:])

        restrict = self.bump(obs - npx.zeros_like(obs).at[2].set(3))
        perturbation = self.tau * restrict * g_out
        input = input.at[0].add(perturbation)

        return PolicyOutput(input)

def create_environment(rng_key, d, pair_first):
    M1, M2 = make_challenging_pair()
    A, K = M1 if pair_first else M2
    model = EmbedEnvironment.create_model()
    vars = model.init(rng_key, npx.zeros((d,)))
    return EmbedEnvironment(A, K, vars, d)

@dataclass
class ExpertDataset(EnvDataset):
    _train_split: Any
    _test_split: Any
    _valid_split: Any

    # the environment parameters
    _A: Array
    _K: Array
    _vars: Any
    _d: int

    def split(self, name):
        if name == "train":
            return SequenceData.from_pytree(self._train_split)
        elif name == "test":
            return SequenceData.from_pytree(self._test_split)
        elif name == "validation":
            return SequenceData.from_pytree(self._valid_split)
    
    def create_env(self, type=None):
        return EmbedEnvironment(self._A, self._K, self._vars, self._d)

def create_data(rng_key, d, N, N_test, T, pair_first):
    env = create_environment(rng_key, d, pair_first=pair_first)
    expert = EmbedExpert(env.K, env.vars, env.tau, env.bump)
    def rollout(rng_key):
        x0_rng, p_rng = foundry.random.split(rng_key)
        x0 = env.reset(x0_rng)
        rollout = foundry.policy.rollout(
            env.step, x0, expert, 
            policy_rng_key=p_rng,
            length=T, last_action=True
        )
        return Step(
            state=rollout.states, 
            reduced_state=rollout.states, 
            observation=rollout.states,
            action=rollout.actions
        )
    train = jax.vmap(rollout)(foundry.random.split(rng_key, N))
    test = jax.vmap(rollout)(foundry.random.split(rng_key, N_test))
    valid = jax.vmap(rollout)(foundry.random.split(rng_key, N_test))

    return  ExpertDataset(train, test, valid, env.A, env.K, env.vars, env.d)


def register_envs(registry : Registry, prefix=None):
    registry.register(
        "lower_bound/stable/1", 
        partial(create_environment, rng_key=foundry.random.key(42), d=2, pair_first=True),
        prefix=prefix
    )
    registry.register(
        "lower_bound/stable/2", 
        partial(create_environment, rng_key=foundry.random.key(42), d=2, pair_first=False),
        prefix=prefix
    )

def register_datasets(registry: Registry, prefix=None):
    registry.register(
        "lower_bound/stable/1",
        partial(create_data, rng_key=foundry.random.key(42), 
                d=4, T=32, N=2048, N_test=64, pair_first=True),
        prefix=prefix
    )
    registry.register(
        "lower_bound/stable/2",
        partial(create_data, rng_key=foundry.random.key(42),
                d=4, T=32, N=2048, N_test=64, pair_first=False),
        prefix=prefix
    )
