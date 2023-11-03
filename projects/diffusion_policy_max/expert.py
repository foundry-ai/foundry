from stanza.runtime import activity
from stanza.rl import ACPolicy
from stanza.rl.nets import MLPActorCritic
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgressBar
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
    trained_params = ppo.train(
        rng_key=PRNGKey(42),
        env=env,
        ac_apply=net.apply,
        init_params=params,
        rl_hooks=[]
    )
    logger.info("Done training PPO expert.")
    exp.add("config", config)
    exp.add("net", net)
    exp.add("fn_params", trained_params.fn_params)
    exp.add("fn_state", trained_params.fn_state)
    exp.add("obs_normalizer", trained_params.obs_normalizer)

def make_expert(repo, env_name, type="ppo", traj_length=None):
    if type == "ppo":
        exp = repo.find(expert=env_name).first()
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
        return policy
    elif type == "mpc":
        env = envs.create(env_name)
        policy = MPC(
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost,
            model_fn=env.step,
            horizon_length=30
        )
        return policy