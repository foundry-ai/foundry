from stanza.dataclasses import dataclass, replace
from stanza.runtime import activity

from stanza.util.logging import logger
from stanza.data.trajectory import Timestep
from stanza.data import Data

from stanza.solver.ilqr import iLQRSolver
from stanza.policies.mpc import MPC, MinimizeMPC

import stanza.util
import stanza.envs as envs
import stanza.policies as policies

import jax
import jax.numpy as jnp

from jax.random import PRNGKey

def centered_log(constraint_fn, x):
    x_flat, x_uf = jax.flatten_util.ravel_pytree(x)[0]
    def log_fn(x_flat):
        x = x_uf(x_flat)
        d = constraint_fn(x)
        return jnp.log(-d)

def make_expert(env_name, env, eta=0.0001, horizon_length=20):
    state_constr, input_constr = None, None
    def constr(states, actions):
        d_s = (
            jnp.reshape(jax.vmap(state_constr)(states), (-1,))
            if state_constr is not None else None
        )
        d_i = (
            jnp.reshape(jax.vmap(input_constr)(actions),(-1,))
            if input_constr is not None else None
        )
        d = jnp.concatenate((d_s, d_i), -1) if d_s is not None and d_i is not None \
            else (d_s if d_s is not None else d_i)
        return d

    def cost_fn(obs, actions):
        # obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        # actions_flat, actions_uf = jax.flatten_util.ravel_pytree(actions)
        # def log_barrier(d, obs_flat, actions_flat):
        #     obs = obs_uf(obs_flat)
        #     actions = actions_uf(actions_flat)
        #     d = constr(obs, actions)
        #     return jnp.log(-d)
        d = constr(obs, actions)
        barrier_cost = jnp.mean(
            -jnp.log(-d)
        ) if d is not None else None
        env_cost = env.cost(obs, actions)
        return eta*barrier_cost + env_cost
    
    def initializer(state0, actions):
        def barrier_cost(obs, actions):
            d = constr(obs, actions)
            # s = jnp.maximum(2*jnp.max(d) + 1e-4, 0)
            s = 0
            s = jax.lax.stop_gradient(s)
            c = -jnp.log(s - d)
            return jnp.mean(c)
        solver = iLQRSolver()
        objective = MinimizeMPC(
            initial_actions=actions,
            state0=state0,
            cost_fn=barrier_cost,
            model_fn=env.step
        )
        res = solver.run(objective)
        # jax.debug.print("{}", res.solution.cost)
        return res.solution.actions

    if env_name == "pendulum":
        input_constr = lambda u: jnp.stack((u - 2, -2 - u))
    elif env_name == "quadrotor":
        input_constr = lambda u: jnp.stack((u[0] - 7.5, -5 - u[0]))
        # state_constr = lambda x: jnp.stack((x.z_dot - 0.5, -0.5 - x.z_dot))
    if state_constr is None and input_constr is None:
        initializer = None
    initializer = None
    mpc = MPC(
        action_sample=env.sample_action(PRNGKey(0)),
        initializer=initializer,
        cost_fn=cost_fn,
        model_fn=env.step,
        horizon_length=horizon_length
    )
    return mpc

def make_rollout(env, policy, 
        traj_length=100, include_jacobian=False):
    def rollout(rng_key):
        x0_rng, policy_rng, env_rng = jax.random.split(rng_key, 3)
        x0 = env.reset(x0_rng)

        def policy_with_jacobian(input):
            if include_jacobian:
                obs_flat, obs_uf = jax.flatten_util.ravel_pytree(input.observation)
                def f(obs_flat):
                    x = obs_uf(obs_flat)
                    out = policy(replace(input, observation=x))
                    action = out.action
                    action_flat = jax.flatten_util.ravel_pytree(action)[0]
                    return action_flat, out
                jac, out = jax.jacobian(f, has_aux=True)(obs_flat)
                out = replace(out, info=replace(out.info, J=jac))
            else:
                out = policy(input)
            return out
        roll = policies.rollout(env.step, 
            x0,
            policy_with_jacobian, length=traj_length,
            policy_rng_key=policy_rng,
            model_rng_key=env_rng,
            observe=env.observe,
            last_input=True)
        return Data.from_pytree(Timestep(
            roll.observations, roll.actions, 
            roll.states, roll.info))
    return rollout

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

@dataclass
class Config:
    env: str = "pendulum"
    batch_size: int = 64
    rng_seed: int = 42
    num_traj: int = 64
    traj_length: int = 100
    include_jacobian: bool = False
    eta: float = 0.0001

@activity(Config)
def generate_data(config, db):
    logger.info(
        f"Generating data for [blue]{config.env}[/blue]"
    )
    data = db.create()
    data.tag(data_for=config.env)
    logger.info(f"Saving data to bucket {data.url}")
    env = envs.create(config.env)
    expert = make_expert(config.env, env)
    rollout_fn = make_rollout(env, expert, 
        config.traj_length,
        config.include_jacobian
    )
    rollout_fn = jax.pmap(rollout_fn, backend="cpu")
    batch_size = min(config.batch_size or config.trajectories, len(jax.devices("cpu")))
    rng = PRNGKey(42)
    rngs = jax.random.split(rng, config.num_traj)
    trajectories = batch_rollout(batch_size, rngs, rollout_fn)

    data.add("trajectories", trajectories)