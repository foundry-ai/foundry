
from functools import partial
from jax.random import PRNGKey
from typing import Any

from stanza.dataclasses import dataclass, replace, field
from stanza.util.logging import logger
from stanza.runtime import activity
from stanza.data import Data, PyTreeData
from stanza.train import Trainer
from stanza.train.rich import RichReporter
from stanza.solver.ilqr import iLQRSolver
from stanza.solver.optax import OptaxSolver
from stanza.solver.newton import NewtonSolver

from stanza.policies.mpc import MPC
from stanza.policies import RandomPolicy, Trajectory
from typing import List

import stanza.policies as policies

import stanza
import stanza.envs
import stanza.util.random

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import sys

import plotly.express as px


_ACTIVATION_MAP = {
    'gelu': jax.nn.gelu,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,
    'tanh': jax.nn.tanh,
}

@dataclass
class Config:
    exp_name: str = None
    learned_model: bool = True
    lr: float = None
    iterations: int = 25000
    batch_size: int = 50
    hidden_dim: int = None
    hidden_layers: int = 3

    # This is trajectories *worth* of data
    # not actual rollout trajectories
    trajectories: List[int] = field(default_factory=lambda: [50, 500, 1000, 2000,
                                        5000, 8000, 10000, 15000, 20000])

    rng_seed: int = 69
    traj_seed: int = 42
    traj_length: int = 25

    use_random: bool = False

    eval_trajs: int = 20

    env: str = "pendulum"

    activation: str = None

    jacobian_regularization: float = 0.0  # 0.0 disables it

    verbose: bool = False
    show_pbar: bool = True

def set_default(config, attr, default):
    if getattr(config, attr) == None:
        setattr(config, attr, default)

def make_solver(gt=False):
    return iLQRSolver()
    if gt:
        return iLQRSolver()
    # use 10000 iterations for the ground truth baseline,
    # but that takes forever, so for the main dataset use 1500 iterations
    # which with ground truth dynamics
    # gets within tiny error of the true solution
    iterations = 5000
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                iterations, alpha=0.1)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-0.2)
    )
    return OptaxSolver(optimizer=optimizer, max_iterations=iterations)

def map_fn(traj):
    states, actions = traj.states, traj.actions
    prev_states = jax.tree_util.tree_map(lambda x: x[:-1], states)
    next_states = jax.tree_util.tree_map(lambda x: x[1:], states)
    return prev_states, actions, next_states

def generate_dataset(config, env, curr_model_fn, rng_key, num_traj, prev_data):
    if config.use_random:
        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
        keys1 = jax.random.split(sk1, num_traj*config.traj_length)
        keys2 = jax.random.split(sk2, num_traj*config.traj_length)
        xs = jax.vmap(env.sample_state)(keys1)
        us = jax.vmap(env.sample_action)(keys2)
        xs_next = jax.vmap(env.step)(xs, us, None)
        data = xs, us, xs_next
        data = Data.from_pytree(data)
        if prev_data is not None:
            data = Data.from_pytree(jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,y)),
                                                data.data, prev_data.data))
        data = data.shuffle(rng_key)
        return data

    if curr_model_fn is None:
        rng_key, sk = jax.random.split(rng_key)
        policy = RandomPolicy(env.sample_action)
    else:
        policy = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=curr_model_fn,
            horizon_length=config.traj_length,
            solver=make_solver(),
            receed=False
            #replan=True
        )
    def rollout(key):
        traj_key, policy_key = jax.random.split(key)
        r = policies.rollout(env.step, env.reset(traj_key), policy,
                                    length=config.traj_length, 
                                    policy_rng_key=policy_key)
        return r.states, r.actions
    rng_key, sk = jax.random.split(rng_key)
    keys = jax.random.split(sk, num_traj)
    keys = keys.reshape((jax.device_count(), -1) + keys.shape[1:])
    output = jax.pmap(jax.vmap(rollout))(keys)
    output = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), output)
    states, actions = output
    earlier_states = jax.tree_util.tree_map(lambda x: x[:,:-1], states)
    later_states = jax.tree_util.tree_map(lambda x: x[:,1:], states)
    data = earlier_states, actions, later_states

    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x,(-1,) + x.shape[2:]), data)
    data = Data.from_pytree(data)
    logger.info("Generated {} new datapoints", data.length)
    if prev_data is not None:
        data = Data.from_pytree(jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,y)),
                                            data.data, prev_data.data))
    logger.info("Dataset contains {} samples", data.length)
    data = data.shuffle(rng_key)
    return data

def evaluate_model(config, env, est_model_fn, traj_key, gt=False):
    policy = MPC(
        action_sample=env.sample_action(PRNGKey(0)),
        cost_fn=env.cost,
        model_fn=est_model_fn,
        horizon_length=config.traj_length,
        solver=make_solver(gt=gt),
        receed=False,
        history=True
    )
    def eval(key):
        r = policies.rollout(env.step, env.reset(key), policy,
                                    length=config.traj_length)
        return r, env.cost(r.states, r.actions), r.final_policy_state
    keys = jax.random.split(traj_key, config.eval_trajs)
    r, c, _ = jax.vmap(eval)(keys)
    r0 = jax.tree_util.tree_map(lambda x: x[0], r)
    logger.info("Eval states r0: {}", r0.states)
    logger.info("Eval actions r0: {}", r0.actions)
    logger.info("Eval costs: {}", c)
    return c

def net_fn(config, x, u):
    activation = _ACTIVATION_MAP[config.activation]
    # default network that does not interpret state values
    u_flat, _ = jax.flatten_util.ravel_pytree(u)
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x)
    mlp = hk.nets.MLP(
        [config.hidden_dim]*config.hidden_layers + [x_flat.shape[0]],
        activation=activation
    )
    input = jnp.concatenate((x_flat, u_flat))
    x_flat = mlp(input) + x_flat
    x = unflatten(x_flat)
    return x

def loss_fn(config, net, model_fn, params, rng_key, sample):
    x, u, x_next = sample
    pred_x = net.apply(params, None, x, u)

    x_next_flat, _ = jax.flatten_util.ravel_pytree(x_next)
    x_pred_flat, _ = jax.flatten_util.ravel_pytree(pred_x)
    loss = jnp.sum(jnp.square(x_next_flat - x_pred_flat))
    stats = {'loss': loss}
    if config.jacobian_regularization > 0:
        true_jac = jax.jacrev(lambda x,u: model_fn(x,u,None), argnums=(0,1))(x, u)
        pred_jac = jax.jacrev(lambda x,u: net.apply(params, None, x, u), argnums=(0,1))(x, u)
        true_jac_flat, _ = jax.flatten_util.ravel_pytree(true_jac)
        pred_jac_flat, _ = jax.flatten_util.ravel_pytree(pred_jac)
        jac_loss = jnp.sum(jnp.square(true_jac_flat - pred_jac_flat))
        loss = loss + config.jacobian_regularization*jac_loss
        stats.update({'jac_loss': jac_loss})
    return loss, stats

def fit_model(config, net, env, dataset, rng_key):
    rng = hk.PRNGSequence(rng_key)
    x_sample, u_sample, _ = dataset.get(dataset.start)
    init_params = net.init(next(rng), x_sample, u_sample)

    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.additive_weight_decay(0.0001),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                config.iterations)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-config.lr))
    trainer = Trainer(
        max_iterations=config.iterations,
        batch_size=config.batch_size,
        optimizer=optimizer
    )
    loss = stanza.partial(loss_fn, config, net, env.step)
    # hold out 100 datapoints for test
    train_data, test_data = PyTreeData.from_data(dataset[100:]), PyTreeData.from_data(dataset[:100])
    with RichReporter(iter_interval=50) as cb:
        res = trainer.train(loss, train_data, 
            rng_key, init_params, hooks=[cb], jit=True)
    learned_model = lambda x, u, rng_key: net.apply(res.fn_params, rng_key, x, u)
    train_loss, train_loss_dict = jax.vmap(partial(loss, res.fn_params, None))(train_data.data)
    logger.info("Train Loss {}, {}", jnp.mean(train_loss), jax.tree_map(lambda x: jnp.mean(x), train_loss_dict))
    test_loss, test_loss_dict = jax.vmap(partial(loss, res.fn_params, None))(test_data.data)
    logger.info("Test Loss {}, {}", jnp.mean(test_loss), jax.tree_map(lambda x: jnp.mean(x), test_loss_dict))
    return learned_model

@activity(Config)
def ilqr_learning(config, database):
    if config.exp_name is not None:
        exp = database.open(config.exp_name)
    else:
        exp = database.open("ilqr").create()
    logger.info(f"Running iLQR learning [blue]{exp.name}[/blue]")
    # set the per-env defaults
    if config.env == "pendulum":
        set_default(config, "lr", 1e-3)
        set_default(config, "hidden_dim", 96)
        set_default(config, "activation", "gelu")  # might want to sweep over gelu/swish
    elif config.env == "quadrotor":
        set_default(config, "lr", 5e-3)
        set_default(config, "hidden_dim", 128)
        set_default(config, "activation", "gelu")  # might want to sweep over gelu/swish

    env = stanza.envs.create(config.env)
    traj_key = stanza.util.random.key_or_seed(config.traj_seed)
    _, eval_key = jax.random.split(traj_key)
    rng_key = stanza.util.random.key_or_seed(config.rng_seed)

    logger.info("Evaluating optimal cost")
    opt_cost = evaluate_model(config, env, env.step, eval_key, gt=True)
    logger.info("Optimal cost: {}", opt_cost)

    # transform the network to a pure function
    net = hk.transform(partial(net_fn, config))
    est_model_fn = None

    # populate the desired trajectories
    data = None
    total_trajs = 0
    samples = []
    subopts = []
    rng = hk.PRNGSequence(rng_key)
    for t in config.trajectories:
        logger.info("Running with {} trajectories", t)
        num_trajs = t - total_trajs
        total_trajs = t
        logger.info("Generating data...")
        data = generate_dataset(config, env, est_model_fn, next(rng), num_trajs, data)
        logger.info("Fitting model...")
        est_model_fn = fit_model(config, net, env, data, next(rng))
        cost = evaluate_model(config, env, est_model_fn, eval_key)
        subopt = (cost - opt_cost)/opt_cost
        logger.info("suboptimality: {}", subopt)
        subopt_m = jnp.mean(subopt)
        subopt_std = jnp.std(subopt)
        logger.info("Cost {}, suboptimality: {} ({})", cost, subopt_m, subopt_std)
        samples.append(total_trajs)
        subopts.append(subopt)

    # Make a plot from the metrics
    samples = jnp.array(samples)
    subopts = jnp.array(subopts)
    exp.log({
        "samples": samples,
        "subopts": subopts
    })
    exp.add('config', config)