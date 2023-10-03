from stanza.util.logging import logger

from stanza.data import PyTreeData
from stanza.data.chunk import chunk_data
from stanza.data.normalizer import LinearNormalizer

from stanza.dataclasses import replace

import stanza.policies as policies

import jax
import jax.numpy as jnp

def load_data(data_db, config):
    logger.info("Reading data...")
    data = data_db.get("trajectories")
    # chunk the data and flatten
    logger.info(f"Chunking {data.length} trajectories")
    val_len = min(data.length/2, 20)
    val_data = PyTreeData.from_data(data[-val_len:], chunk_size=64)
    data = data[:-val_len]
    if config.num_trajectories is not None:
        data = data[:config.num_trajectories]
    data = PyTreeData.from_data(data, chunk_size=64)

    data_flat = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    normalizer = LinearNormalizer.from_data(data_flat)

    val_data_pt = val_data.data.data
    val_trajs = policies.Rollout(
        states=val_data_pt.state,
        actions=val_data_pt.action,
        observations=val_data_pt.observation,
        info=val_data_pt.info
    )
    def chunk(traj):
        # Throw away the state, we use only
        # the observations and actions
        traj = traj.map(lambda x: replace(x, state=None))
        traj = chunk_data(traj,
            chunk_size=config.diffusion_horizon, 
            start_padding=config.obs_horizon - 1,
            end_padding=config.action_horizon - 1)
        return traj
    data = data.map(chunk)
    val_data = val_data.map(chunk)

    # Load the data into a PyTree
    data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    val_data = PyTreeData.from_data(val_data.flatten(), chunk_size=4096)
    logger.info("Data size: {}",
        sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size*x.itemsize, data.data))))
    logger.info("Data Loaded! Computing normalizer")
    logger.info("Normalizer computed")
    return data, val_data, val_trajs, normalizer

def eval(val_trajs, env, policy, rng_key):
    x0s = jax.tree_map(lambda x: x[:,0], val_trajs.states)
    N = jax.tree_util.tree_leaves(val_trajs.states)[0].shape[0]
    length = jax.tree_util.tree_leaves(val_trajs.states)[0].shape[1]
    def roll(x0, rng):
        model_rng, policy_rng = jax.random.split(rng)
        return policies.rollout(env.step, x0,
            observe=env.observe, policy=policy, length=length, 
            model_rng_key=model_rng, policy_rng_key=policy_rng,
            last_input=True
        )
    roll = jax.vmap(roll)
    rngs = jax.random.split(rng_key, N)
    rolls = roll(x0s, rngs)
    from stanza.util import extract_shifted

    state_early, state_late = jax.vmap(extract_shifted)(rolls.states)
    actions = jax.tree_map(lambda x: x[:,:-1], rolls.actions)
    vreward = jax.vmap(jax.vmap(env.reward))
    policy_r = vreward(state_early, actions, state_late)
    policy_r = jnp.sum(policy_r, axis=1)

    state_early, state_late = jax.vmap(extract_shifted)(val_trajs.states)
    actions = jax.tree_map(lambda x: x[:,:-1], val_trajs.actions)
    expert_r = vreward(state_early, actions, state_late)
    expert_r = jnp.sum(expert_r, axis=1)
    reward_ratio = policy_r / expert_r
    return rolls, jnp.mean(reward_ratio)