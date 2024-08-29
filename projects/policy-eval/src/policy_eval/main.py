import foundry.core as F
import foundry.numpy as jnp

from foundry.datasets.env import datasets

import foundry.random

from foundry.random import PRNGSequence
from foundry.env import ImageActionsRender
from foundry.train.reporting import Video

from .config import Config
from .common import Inputs, MethodConfig

import functools
import logging

logger = logging.getLogger(__name__)

@F.jit
def policy_rollout(env, T, 
                    x0, rng_key, policy, render_video):
    r_policy, r_env = foundry.random.split(rng_key)
    rollout = foundry.policy.rollout(
        env.step, x0,
        policy, observe=env.boserve,
        model_rng_key=r_env,
        policy_rng_key=r_policy,
        length=T
    )
    pre_states, actions, post_states = (
        jax.tree.map(lambda x: x[:-1], rollout.states),
        jax.tree.map(lambda x: x[:-1], rollout.actions),
        jax.tree.map(lambda x: x[1:], rollout.states)
    )
    rewards = jax.vmap(env.reward)(
        pre_states, actions, post_states
    )
    # max over time for the rewards
    reward = jnp.max(rewards, axis=0)
    return rollout, reward

@F.jit
def render_video(env, render_width, 
                render_height, rollout):
    states = rollout.states
    actions = rollout.infos
    return F.vmap(env.render)(states, ImageActionsRender(
        render_width, render_height,
        actions
    ))

@F.jit
def validate(env, T, render_width, render_height,
                videos, x0s, rng_key, policy):
    rollout_fn = partial(policy_rollout, env, T) 
    render_fn = partial(render_video, env, 
        render_width, render_height
    )
    N = tree.axis_size(x0s, 0)
    rngs = foundry.random.split(rng_key, N)

    rollouts, rewards = F.vmap(rollout_fn)(x0s, rngs)

    if videos is None:
        return rewards
    # render the videos
    video_rollouts = tree.map(
        lambda x: x[:videos], rollouts
    )
    jax.vmap(render_fn)()


def process_data(config, env, data):
    logger.info("Computing full state from data...")

    def process_element(element):
        return env.full_state(element.reduced_state)
    data = data.map_elements(process_element).cache()

    logger.info("Chunking data...")

    data = data.chunk(
        config.action_length + config.obs_length
    )
    def process_chunk(chunk):
        states = chunk.elements
        actions = jax.vmap(lambda s: env.observe(s, config.action_config))(states)
        actions = jax.tree.map(lambda x: x[-config.action_length:], actions)
        obs_states = jax.tree.map(lambda x: x[:config.obs_length], states)
        curr_state = jax.tree.map(lambda x: x[-1], obs_states)
        obs = jax.vmap(env.observe)(obs_states)
        return Sample(
            curr_state, obs, actions
        )

    data = data.map(process_chunk)

    return data

def run(config: Config):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running {config}")

    # ---- set up the random number generators ----

    # split the master RNG into train, eval
    train_key, eval_key = foundry.random.split(foundry.random.key(config.seed))

    # fold in the seeds for the train, eval
    train_key = foundry.random.fold_in(train_key, config.train_seed)
    eval_key = foundry.random.fold_in(eval_key, config.eval_seed)

    # ---- set up the training data -----

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = datasets.create(config.dataset)
    env = dataset.create_env()

    rollout = functools.partial(
        policy_rollout, config.timesteps
    )

    train_data = dataset.splits["train"]
    if config.train_trajectories is not None:
        train_data = train_data.slice(0, config.train_trajectories)

    test_data = dataset.splits["test"]
    if config.train_trajectories is not None:
        test_data = test_data.slice(0, config.test_trajectories)

    # validation trajectories
    # are used for
    validation_data = dataset.splits["validation"]
    if config.validation_trajectories is not None:
        test_data = test_data.slice(0, config.validation_trajectories)

    method = config.method_config

    inputs = Inputs(
        timesteps = config.timesteps,
        rng=PRNGSequence(train_key)
    )
    final_policy = method.run(inputs)