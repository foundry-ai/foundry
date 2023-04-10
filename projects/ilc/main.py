import jax
import jax.numpy as jnp
import jinx.envs as envs
import jinx.random

from jax.random import PRNGKey

from dataclasses import dataclass, field
from typing import Any, Dict
from jinx.logging import logger

from jinx.policy.iterative import FeedbackMPC
from jinx.policy.grad_estimator import IsingEstimator
from jinx.trainer import Trainer

from jinx.experiment.runtime import activity
from jinx.experiment import Figure, Video

from functools import partial

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
    traj_seed: int = 42

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

@activity('iterative_learning', Config)
def iterative_learning(config, exp=None):
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

    env = envs.create(config.env_type)
    rng_key = jinx.random.key_or_seed(config.rng_seed)
    traj_key = jinx.random.key_or_seed(config.traj_seed)

    model_fn = env.step
    cost_fn = env.cost
    traj_cost_fn = partial(envs.trajectory_cost, env.cost)

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

    policy = FeedbackMPC(
        env.reset(PRNGKey(0)),
        env.sample_action(PRNGKey(0)), 
        env.cost, model_fn,
        config.horizon_length,
        optimizer=optimizer,
        iterations=config.iterations,
        use_gains=config.use_gains,

        burn_in=config.burn_in,
        Q_coef=config.Q_coef,
        R_coef=config.R_coef,

        receed=config.receed,
        grad_estimator=est if config.estimate_grad else None,
    )
    x0 = env.reset(traj_key)
    logger.info(f"Running {config}")

    # Rollout with the true environment dynamics!
    pol_state, states, us = envs.rollout_policy(env.step, x0,
                                config.traj_length, policy,
                                ret_policy_state=True)
    traj_cost = traj_cost_fn(states, us)
    logger.info("Final State: {}", jax.tree_map(lambda x: x[-1], states))
    logger.info("Final Cost: {}", traj_cost)
    
    optim_history = pol_state.optim_history
    cost_history = optim_history.cost
    est_state_history = optim_history.est_state
    iterations = optim_history.iteration[-1]

    logger.info("{} iterations", iterations)

    # output to experiment
    for i in range(iterations):
        exp.log({'cost': cost_history[i],
            'grad_norm': optim_history.grad_norm[i],
            'samples': est_state_history.total_samples[i] if est_state_history else None,
        })

    # make a plot of the trajectory, video of pendulum
    vis = env.visualize(states, us)
    exp.log(vis)

    # serialize to results for our own records
    results = Results(
        xs=states,
        us=us,
        config=config,
        iterations=iterations,
        final_cost=cost_history[-1],
        cost_history=cost_history,
        sample_history=est_state_history.total_samples if est_state_history is not None else None
    )
    if config.save_file is not None:
        root_dir = os.path.abspath(os.path.join(__file__,'..','..'))
        with open(os.path.join(root_dir, 'results', config.save_file), 'wb') as f:
            pickle.dump(results, f)