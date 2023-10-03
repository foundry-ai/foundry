from stanza.util.logging import logger

from stanza.data import PyTreeData
from stanza.data.chunk import chunk_data
from stanza.data.normalizer import LinearNormalizer

from stanza.dataclasses import replace

import stanza.policies as policies
import stanza.util

import jax
import jax.numpy as jnp

def load_data(data_db, num_trajectories,
              obs_horizon, action_horizon, action_padding):
    logger.info("Reading data...")
    data = data_db.get("trajectories")
    # chunk the data and flatten
    logger.info(f"Chunking {data.length} trajectories")
    val_len = min(data.length/2, 20)
    val_data = PyTreeData.from_data(data[-val_len:], chunk_size=64)
    data = data[:-val_len]
    if num_trajectories is not None:
        data = data[:num_trajectories]
    data = PyTreeData.from_data(data, chunk_size=64)
    data_flat = data.flatten().map(lambda x: replace(x, info=replace(x.info, knn=None)))
    data_flat = PyTreeData.from_data(data_flat, chunk_size=4096)
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
            chunk_size=obs_horizon + action_horizon + action_padding - 1,
            start_padding=obs_horizon - 1,
            end_padding=action_horizon - 1)
        def slice_chunk(x):
            obs = jax.tree_map(lambda x: x[:obs_horizon], x.observation)
            return replace(x, observation=obs,
                            info=replace(x.info, knn=None)
                           )
        traj = traj.map(slice_chunk)
        return traj
    data = data.map(chunk)
    val_data = val_data.map(chunk)
    # Load the data into a PyTree
    data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    val_data = PyTreeData.from_data(val_data.flatten(), chunk_size=4096)
    logger.info("Data size: {}",
        sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size*x.itemsize, data.data))))
    return data, val_data, val_trajs, normalizer

def knn_data(data, n):
    data = data.data
    # make a distance matrix
    def nearest(x):
        dists = jax.vmap(stanza.util.l2_norm_squared, in_axes=(0, None))(
            data.observation, x.observation
        )
        _, indices = jax.lax.top_k(dists, n + 1)
        indices = indices[1:]
        neigh = jax.tree_map(lambda x: jnp.take(x, indices, 
                                unique_indices=True, axis=0), data)
        neigh = replace(neigh, info=None)
        return neigh
    # find the nearest for every data point
    knn = stanza.util.map(nearest, vsize=128)(data)
    data = replace(
        data, info=replace(data.info, knn=knn)
    )
    return PyTreeData(data)

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