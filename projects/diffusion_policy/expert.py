from stanza.runtime import activity
from stanza.rl import ACPolicy
from stanza.rl.nets import MLPActorCritic
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgress
from stanza import Partial
from stanza.util.logging import logger
from stanza.rl.ppo import PPO
from stanza.train import Trainer
from stanza.data.trajectory import Timestep
from stanza.data import Data

from stanza.policies.mpc import MPC
from stanza.solver.ilqr import iLQRSolver
from stanza.solver.optax import OptaxSolver

import optax

import stanza.envs as envs
import stanza.policies as policies
import functools
import jax
import jax.numpy as jnp

from jax.random import PRNGKey
from stanza.dataclasses import dataclass, replace

@dataclass
class ExpertConfig:
    env_name: str = "brax/ant"
    expert_name: str = None
    episode_length: int = 1000
    total_timesteps: int = 10_000_000

@activity(ExpertConfig)
def train(config, db):
    if config.expert_name is None:
        config = replace(config, expert_name=config.env_name.replace("/","_"))
    exp = db.open("experts").open(config.expert_name)
    logger.info(f"Logging results to {exp.name}")
    env = envs.create(config.env_name)

    net = MLPActorCritic(env.sample_action(PRNGKey(0)))
    params = net.init(PRNGKey(42),
        env.observe(env.sample_state(PRNGKey(0))))

    display = ConsoleDisplay()
    display.add("ppo", StatisticsTable(), interval=1)
    display.add("ppo", LoopProgress("RL"), interval=1)

    ppo = PPO(
        episode_length=config.episode_length,
        total_timesteps=config.total_timesteps,
        trainer = Trainer(
            optimizer=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(3e-4, eps=1e-5)
            )
        )
    )
    logger.info("Training PPO expert...")
    with display as dh:
        trained_params = ppo.train(
            rng_key=PRNGKey(42),
            env=env,
            ac_apply=net.apply,
            init_params=params,
            rl_hooks=[dh.ppo]
        )
    logger.info("Done training PPO expert.")
    exp.add("config", config)
    exp.add("net", net)
    exp.add("fn_params", trained_params.fn_params)
    exp.add("fn_state", trained_params.fn_state)
    exp.add("obs_normalizer", trained_params.obs_normalizer)

@dataclass
class GenerateConfig:
    expert_name: str = None
    mpc_env: str = None
    data_name: str = None
    traj_length: int = 1000
    trajectories: int = 1000
    batch_size: int = None
    rng_seed: int = 42
    include_jacobian: bool = True

def make_expert(config, db):
    if config.expert_name is not None:
        exp = db.open(f"experts/{config.expert_name}")
        if not "config" in exp:
            print(exp.children)
            logger.error("Expert not found")
            return
        expert_config = exp.get("config")
        expert_net = exp.get("net")
        expert_params = exp.get("fn_params")
        expert_state = exp.get("fn_state")
        normalizer = exp.get("obs_normalizer")

        if config.data_name is None:
            config = replace(config, data_name=config.expert_name)

        ac_apply = expert_net.apply
        ac_apply = Partial(ac_apply, expert_params)
        policy = ACPolicy(ac_apply, normalizer, use_mean=True)
        return expert_config.env_name, policy
    elif config.mpc_env is not None:
        env = envs.create(config.mpc_env)
        policy = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=env.step,
            horizon_length=config.traj_length,
        )
        return config.mpc_env, policy

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
    return jax.tree_map(lambda x: jnp.concatenate(x, axis=0), trajactories)

@activity(GenerateConfig)
def generate_data(config, db):
    if config.data_name is None and config.expert_name is not None:
        config = replace(config, data_name=config.expert_name)
    if config.data_name is None and config.mpc_env is not None:
        config = replace(config, data_name=config.mpc_env)

    env_name, policy = make_expert(config, db)
    env = envs.create(env_name)
    data = db.open(f"expert_data/{config.data_name}")
    logger.info(f"Saving to {data.name}")

    def rollout(rng_key):
        x0_rng, policy_rng, env_rng = jax.random.split(rng_key, 3)

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
            env.reset(x0_rng), 
            policy_with_jacobian, length=config.traj_length,
            policy_rng_key=policy_rng,
            model_rng_key=env_rng,
            observe=env.observe,
            last_input=True)
        return Data.from_pytree(Timestep(
            roll.observations, roll.actions, 
            roll.states, roll.info))
    
    rngs = jax.random.split(PRNGKey(config.rng_seed), config.trajectories)
    if config.mpc_env is not None:
        batch_size = min(config.batch_size or config.trajectories, len(jax.devices("cpu")))
        trajectories = batch_rollout(batch_size,
                rngs, jax.pmap(rollout, backend="cpu"))
    else:
        trajectories = batch_rollout(batch_size,
                rngs, jax.vmap(rollout))
    logger.info("Done generating data, saving to output...")
    trajectories = Data.from_pytree(trajectories)
    data.add("env_name", env_name)
    data.add("trajectories", trajectories)