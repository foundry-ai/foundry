import jinx.envs as envs
import jax.numpy as jnp
from jax.random import PRNGKey

# def test_gym():
#     env = envs.create('gym', 'Humanoid-v3')
#     state = env.reset(PRNGKey(0))
#     state = env.step(state, jnp.zeros((env.action_size,)))
#     x = env.observe(state, 'x')

def test_brax():
    env = envs.create('brax', 'humanoid')
    state = env.reset(PRNGKey(0))
    state = env.step(state, jnp.zeros((env.action_size,)))
    x = state.x