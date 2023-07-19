from stanza.runtime import activity
from stanza.runtime.reporting import Figure

from stanza.policies.mpc import BarrierMPC
from stanza.policies.imitation_learning import ImitationLearning

from stanza.dataset.env import EnvDataset
from stanza.dataset import PyTreeDataset

from stanza.util.logging import logger
from stanza.dataclasses import dataclass, replace

import stanza.policies
import stanza.envs as envs

from jax.random import PRNGKey

from functools import partial

import haiku as hk

import jax
import jax.numpy as jnp
import optax

import plotly.graph_objects as go

@dataclass
class Config:
    seed: int = 42

    traj_length: int = 31
    horizon_length: int = 20
    trajectories: int = 20

    jac_lambda: float = 0.1
    barrier_eta: float = 0.01

    # learning parameters
    learning_rate = 0.001
    weight_decay = 0.01
    batch_size = 30
    epochs: int = 1000

    env_type: str = "linear"
    env_name: str = None

def net_fn(config, u_sample, x):
    u_flat, unflatten = jax.flatten_util.ravel_pytree(u_sample)
    mlp = hk.nets.MLP([60, 60, 60, u_flat.shape[0]], activation=jax.nn.gelu)

    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    u_flat = mlp(x_flat)

    return unflatten(u_flat)

def run_learning(config, rng_key, dataset, env, policy, net):
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.additive_weight_decay(config.weight_decay),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                config.epochs*config.trajectories*
                                config.traj_length / config.batch_size)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-config.learning_rate))

    il = ImitationLearning(net, epochs=config.epochs,
        jac_lambda=config.jac_lambda, optimizer=optimizer,
        batch_size=config.batch_size)

    rng_key, sk = jax.random.split(rng_key)
    imitator = il.run(sk, dataset)
    # roll out the policy and the imitator on the same rng
    x0s = jax.vmap(env.reset)(jax.random.split(rng_key, 20))

    rollout_policy = partial(stanza.policies.rollout, env.step, policy=policy, length=config.traj_length)
    rollout_imitator = partial(stanza.policies.rollout, env.step, policy=imitator, length=config.traj_length)

    policy_xs, policy_us = jax.vmap(rollout_policy)(x0s)
    imitator_xs, imitator_us = jax.vmap(rollout_imitator)(x0s)

    # vmap distance function over both initial condition and time axes
    # mapped_dist = jax.vmap(jax.vmap(stanza.util.ravel_dist, in_axes=(0,0)), in_axes=(0,0))
    # x_dist = jnp.mean(jnp.mean(mapped_dist(policy_xs, imitator_xs), axis=1))
    # u_dist = jnp.mean(jnp.mean(mapped_dist(policy_us, imitator_us), axis=1))
    # return policy_xs, imitator_xs, x_dist, u_dist

@activity(Config)
def learn_mpc(config, run):
    env = env.create(config.env_type)

    net = hk.transform_with_state(partial(net_fn, 
                config, env.sample_action(PRNGKey(0))))

    policy = BarrierMPC(
        env.reset(PRNGKey(0)), env.sample_action(PRNGKey(0)),
        env.cost, env.step,
        config.horizon_length,
        barrier_sdf=env.barrier if hasattr(env, 'barrier') else None,
        barrier_eta=config.barrier_eta
    )

    rng_key = PRNGKey(config.seed)

    sk, rng_key = jax.random.split(rng_key)
    dataset = EnvDataset(sk, env, config.traj_length, policy)
    dataset = dataset[:config.trajectories]

    logger.info("Constructing in-memory dataset")
    dataset = PyTreeDataset.from_dataset(dataset)

    policy_xs, bc_xs, bc_x_dist, bc_u_dist = run_learning(config, rng_key, dataset, env, policy, net)

    logger.info(f"x_dist: {bc_x_dist}, u_dist: {bc_u_dist}")

    policy_xs = jax.tree_util.tree_map(lambda x: x[0], policy_xs)
    bc_xs = jax.tree_util.tree_map(lambda x: x[0], bc_xs)

    if config.env_type == "pendulum":
        fig = go.Figure()
        fig.add_trace(go.Scatter(name='gt', x=jnp.squeeze(policy_xs.angle,-1), y=jnp.squeeze(policy_xs.vel,-1)))
        fig.add_trace(go.Scatter(name='learned', x=jnp.squeeze(bc_xs.angle,-1), y=jnp.squeeze(bc_xs.vel,-1)))
        fig.update_layout(xaxis_title="Theta", yaxis_title="Theta Dot")
        run.log({'traj': Figure(fig)})
    elif config.env_type == "linear":
        fig = go.Figure()
        fig.add_trace(go.Scatter(name='gt', x=policy_xs[:,0], y=policy_xs[:,1]))
        fig.add_trace(go.Scatter(name='learned', x=bc_xs[:,0], y=bc_xs[:,1]))
        fig.update_layout(xaxis_title="x0", yaxis_title="x1")
        run.log({'traj': Figure(fig)})
    logger.info("Done!")

@activity('compare_etas', Config)
def compare_etas(config, run):
    env_args = {'name': config.env_name} if config.env_name is not None else {}
    env = env.create(config.env_type, **env_args)

    net = hk.transform_with_state(partial(net_fn, 
                config, env.sample_action(PRNGKey(0))))

    etas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    bc_x_dists = []
    tasil_x_dists = []
    for eta in etas:
        policy = BarrierMPC(
            env.sample_action(PRNGKey(0)),
            env.cost, env.step,
            config.horizon_length,
            barrier_sdf=env.barrier if hasattr(env, 'barrier') else None,
            barrier_eta=eta
        )
        rng_key = PRNGKey(config.seed)

        sk, rng_key = jax.random.split(rng_key)
        dataset = EnvDataset(sk, env, config.traj_length, policy, jacobians=True)
        dataset = dataset[:config.trajectories]
        logger.info("Building dataset for eta={}", eta)
        dataset = dataset.read()

        # Do vanilla behavior cloning
        logger.info("Running tasil cloning")
        _, _, tasil_x_dist, tasil_u_dist = run_learning(config, rng_key, dataset, env, policy, net)
        logger.info('compare_tasil', f"x_dist: {tasil_x_dist}, u_dist: {tasil_u_dist}")
        tasil_x_dists.append(tasil_x_dist)

        logger.info("Running behavior cloning")
        bc_config = replace(config, jac_lambda=0)
        _, _, bc_x_dist, bc_u_dist = run_learning(bc_config, rng_key, dataset, env, policy, net)
        logger.info(f"x_dist: {bc_x_dist}, u_dist: {bc_u_dist}")
        bc_x_dists.append(bc_x_dist)
    
    # make the mega plot!
    fig = go.Figure()
    fig.add_trace(go.Scatter(name='tasil', x=etas, y=tasil_x_dists))
    fig.add_trace(go.Scatter(name='bc', x=etas, y=bc_x_dists))
    fig.update_layout(xaxis_title="Eta", yaxis_title="Mean Test L2 State Dist")
    run.log({'traj': Figure(fig)})
    logger.info("Done!")

@activity('compare_tasil', Config)
def compare_tasil(config, run):
    env = env.create(config.env_type)

    net = hk.transform_with_state(partial(net_fn, 
                config, env.sample_action(PRNGKey(0))))

    policy = BarrierMPC(
        env.sample_action(PRNGKey(0)),
        env.cost, env.step,
        config.horizon_length,
        barrier_sdf=env.barrier if hasattr(env, 'barrier') else None,
        barrier_eta=config.barrier_eta
    )

    # Do vanilla behavior cloning
    logger.info(f"Running tasil cloning")
    policy_xs, tasil_xs, tasil_x_dist, tasil_u_dist = run_learning(config, env, policy, net)
    logger.info(f"x_dist: {tasil_x_dist}, u_dist: {tasil_u_dist}")

    logger.info(f"Running behavior cloning")
    bc_config = replace(config, jac_lambda=0)
    policy_xs, bc_xs, bc_x_dist, bc_u_dist = run_learning(bc_config, env, policy, net)
    logger.info(f"x_dist: {bc_x_dist}, u_dist: {bc_u_dist}")
    logger.info("Making graph...")
    logger.info("Done!")

@activity(Config)
def solve_traj(config, run):
    env_args = {'name': config.env_name} if config.env_name is not None else {}
    if config.env_type == 'pendulum':
        env_args['sub_steps'] = 5
    env = env.create(config.env_type, **env_args)

    policy = BarrierMPC(
        env.sample_action(PRNGKey(0)),
        env.cost, env.step,
        config.horizon_length,
        barrier_sdf=env.barrier if hasattr(env, 'barrier') else None,
        barrier_eta=config.barrier_eta
    )
    x0 = env.reset(jnp.array([3164236999, 3984487275], dtype=jnp.uint32))
    #x0 = env.reset(PRNGKey(config.seed))
    logger.info("u0: {}", policy(x0))
    xs, us, jacs = stanza.policies.rollout(env.step, env.reset(PRNGKey(config.seed)), 20, policy, jacobians=True)
    logger.info('learn_mpc', "  us: {}", us)
    logger.info('learn_mpc', "  xs: {}", jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(xs))
    logger.info("jacs: {}", jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(jacs))