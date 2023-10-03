from stanza.dataclasses import dataclass
from stanza.runtime import activity
from stanza.util.logging import logger

import stanza.envs as envs

from diffusion_policy.util import load_data

import jax

@dataclass
class Config:
    data: str

@activity(Config)
def train_policy(config, database):
    exp = database.open("diffusion_policy").create()
    logger.info("Train policy")

    data_db = database.open(f"expert_data/{config.data}")
    env_name = data_db.get("env_name")
    env = envs.create(env_name)
    # load the per-env defaults into config
    logger.info("Using environment [blue]{}[/blue] with config: {}", env_name, config)
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, val_trajs, normalizer = load_data(data_db, config)
    # move to GPU
    data, val_data, val_trajs, normalizer = jax.device_put(
        (data, val_data, val_trajs, normalizer), device=jax.devices("gpu")[0])
    logger.info("Dataset size: {} chunks", data.length)