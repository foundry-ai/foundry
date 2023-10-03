from stanza.runtime import activity
from stanza.util.logging import logger
from stanza.dataclasses import dataclass, replace
from stanza.data import Data, PyTreeData
from stanza.data.normalizer import LinearNormalizer
from stanza.data.trajectory import Timestep
from stanza.data.chunk import chunk_data

from diffusion_policy.expert import make_expert

import stanza.envs as envs

import jax
import jax.numpy as jnp

import stanza.policies as policies

from jax.random import PRNGKey

@dataclass
class GenerateConfig:
    expert_name: str = None
    env: str = None
    policy: str = "mpc"
    traj_length: int = 1000
    trajectories: int = 1000
    batch_size: int = None
    rng_seed: int = 42
    include_jacobian: bool = True

@activity(GenerateConfig)
def generate_data(config, repo):
    policy = make_expert(repo, config.env, config.policy,
                         traj_length=config.traj_length)
    env = envs.create(config.env)
    # create a data bucket
    data = repo.create()
    # tag this as data for a given environment
    data.tag(data_for=config.env)
    logger.info(f"Saving data to bucket {data.url}")

    def rollout(rng_key):
        x0_rng, policy_rng, env_rng = jax.random.split(rng_key, 3)
        x0 = env.sample_state(x0_rng) 

        def policy_with_jacobian(input):
            if config.include_jacobian:
                obs_flat, obs_uf = jax.flatten_util.ravel_pytree(input.observation)
                def f(obs_flat):
                    x = obs_uf(obs_flat)
                    out = policy(replace(input, observation=x))
                    action = out.action
                    action_flat = jax.flatten_util.ravel_pytree(action)[0]
                    return action_flat, out
                jac, out = jax.jacobian(f, has_aux=True)(obs_flat)
                out = replace(out, info=replace(out.info, K=jac))
                return out
            else:
                return policy(input)
        roll = policies.rollout(env.step, 
            x0,
            policy_with_jacobian, length=config.traj_length,
            policy_rng_key=policy_rng,
            model_rng_key=env_rng,
            observe=env.observe,
            last_input=True)
        return Data.from_pytree(Timestep(
            roll.observations, roll.actions, 
            roll.states, roll.info))
    
    rngs = jax.random.split(PRNGKey(config.rng_seed), config.trajectories)
    if config.policy == "mpc":
        batch_size = min(config.batch_size or config.trajectories, len(jax.devices("cpu")))
        trajectories = batch_rollout(batch_size,
                rngs, jax.pmap(rollout, backend="cpu"))
    else:
        trajectories = batch_rollout(batch_size,
                rngs, jax.vmap(rollout))
    logger.info("Done generating data, saving to output...")
    trajectories = Data.from_pytree(trajectories)
    data.add("trajectories", trajectories)

def load_data(data_db, config):
    logger.info("Reading data...")
    data = data_db.get("trajectories")
    # chunk the data and flatten
    logger.info("Chunking trajectories")
    val_data = PyTreeData.from_data(data[-20:], chunk_size=64)
    data = data[:-20]
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

def batch_rollout(batch_size, rng_keys, rollout_fn):
    N = rng_keys.shape[0]
    batch_size = batch_size or N
    num_batches = ((N - 1) // batch_size) + 1
    logger.info(f"Generating {num_batches} batches of size {batch_size}")
    trajactories = []
    for i in range(num_batches):
        logger.info(f"Generating batch {i}: {i*batch_size} completed")
        rng_batch = rng_keys[i*batch_size:(i+1)*batch_size]
        traj_batch = rollout_fn(rng_batch)
        # wait for the previous batch to finish
        # before generating more
        if len(trajactories) > 1:
            jax.block_until_ready(trajactories[-1])

        trajactories.append(traj_batch)
    return jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *trajactories)