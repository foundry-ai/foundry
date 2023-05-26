import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import stanza.envs as envs
import stanza.policies as policies
import stanza.util.random as random

from stanza.runtime import activity
from stanza.solver.optax import OptaxSolver
from stanza.policies.mpc import MPC
from stanza.policies.iterative import EstimatorRollout, FeedbackRollout
from stanza.util.logging import logger
from stanza.util.grad_estimator import IsingEstimator

from functools import partial
from dataclasses import dataclass, field
from typing import Any, Dict

import haiku as hk

import numpy as np
import optax
import plotly.express as px
import os
import pickle

@dataclass
class Config:
    # Estimate the gradients or learn
    # the true gradients
    estimate_grad: bool = False
    samples : int = None
    sigma : float = None

    learning_rate : float = None
    decay_iterations : int = None
    decay_alpha : float = None
    b1: float = None
    b2: float = None

    receed: bool = False
    rng_seed: int = 42

    traj_length: int = None
    horizon_length: int = None
    iterations: int = None

    # If specified, overrides iterations
    # such that iterations*samples <= trajectories
    trajectories: int = None

    use_gains: bool = False
    burn_in: int = None
    Q_coef: float = None
    R_coef: float = None

    env_type: str = "pendulum"
    save_file: str = None

@dataclass
class Results:
    config: Config
    cost_history: np.ndarray
    sample_history: np.ndarray
    iterations: int
    final_cost: float
    xs: Any
    us: Any

# ---------------------- For dynamics-learning relevant code ---------------------
def set_default(config, attr, default):
    if getattr(config, attr) == None:
        setattr(config, attr, default)

@activity(Config)
def iterative_learning(config, database):
    exp = database.open("iterative_learning").create()
    logger.info(f"Running iterative learning [blue]{exp.name}[/blue]")
    # set the per-env defaults for now
    if config.env_type == "pendulum":
        set_default(config, 'traj_length', 50)
        set_default(config, 'horizon_length', 50)

        set_default(config, 'iterations', 1000)
        set_default(config, 'learning_rate', 0.2)
        set_default(config, 'b1', 0.9)
        set_default(config, 'b2', 0.999)

        set_default(config, 'decay_iterations', 500)
        set_default(config, 'decay_alpha', 0.1)
        set_default(config, 'samples', 30)
        set_default(config, 'sigma', 0.1)
        set_default(config, 'burn_in', 10)
        set_default(config, 'Q_coef', 0.1)
        set_default(config, 'R_coef', 1)
    elif config.env_type == "quadrotor":
        set_default(config, 'traj_length', 50)
        set_default(config, 'horizon_length', 50)

        set_default(config, 'iterations', 200)
        set_default(config, 'learning_rate', 0.01)
        # let's screww around with the other adam params
        set_default(config, 'b1', 0.8)
        set_default(config, 'b2', 0.999)

        set_default(config, 'decay_iterations', 1000)
        set_default(config, 'decay_alpha', 0.01)
        set_default(config, 'samples', 50)
        set_default(config, 'sigma', 0.1)
        set_default(config, 'burn_in', 10)
        set_default(config, 'Q_coef', 0.1)
        set_default(config, 'R_coef', 1)

    if config.trajectories:
        config.iterations = config.trajectories / config.samples
    logger.info(f"Config: {config}")

    env = envs.create(config.env_type)
    rng_key = random.key_or_seed(config.rng_seed)
    rng_key, traj_key = jax.random.split(rng_key)

    if not config.receed and config.horizon_length < config.traj_length:
        logger.warn("Receeding horizon disabled, increasing horizon to trajectory length")
        config.horizon_length = config.traj_length

    est = IsingEstimator(
        rng_key, config.samples, config.sigma
    )
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                config.decay_iterations, alpha=config.decay_alpha)),
        optax.scale_by_adam(b1=config.b1, b2=config.b2, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-config.learning_rate)
    )
    roller = FeedbackRollout(
        model_fn=env.step,
        burn_in=config.burn_in, 
        Q_coef=config.Q_coef, R_coef=config.R_coef
    ) if config.use_gains else EstimatorRollout(
        model_fn=env.step, grad_estimator=None
    )

    policy = MPC(
        action_sample=env.sample_action(PRNGKey(0)), 
        cost_fn=env.cost, 
        rollout_fn=roller,
        horizon_length=config.horizon_length,
        solver=OptaxSolver(optimizer=optimizer,
                max_iterations=config.iterations),
        receed=config.receed,
        history=True
    )

    x0 = env.reset(traj_key)
    def eval(rng_key):
        rollout = policies.rollout(env.step, x0,
                                    policy, length=config.traj_length)
        traj_cost = env.cost(rollout.states, rollout.actions)
        return rollout, traj_cost
    keys = jax.random.split(traj_key, config.eval_trajs)
    rollouts, costs = jax.vmap(eval)(keys)

    # Rollout with the true environment dynamics!
    logger.info("Final Cost: {}", costs)

    # output to experiment
    # exp.log({
    #     'cost': solver_history.cost[:iterations],
    #     'samples': est_state_history.total_samples[:iterations] if est_state_history else None,
    # })

    # # make a plot of the trajectory, video of pendulum
    # vis = env.visualize(rollout.states, rollout.actions)
    # exp.log(vis)