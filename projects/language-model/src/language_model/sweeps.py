from foundry.runtime import ConfigProvider, command, setup
from functools import partial
from ray import tune
from ray.air.integrations.wandb import setup_wandb

import logging
logger = logging.getLogger(__name__)

def sam_start_points_run(config, sweep_config: dict):
    logger.info(f"Running sweep parameters {sweep_config}")
    from .core import train
    # initialize the wandb sweep!
    run = setup_wandb(config)
    train(run, config)

def sam_start_points_sweep(config: ConfigProvider):
    from .core import Config
    base_config = Config.parse(config)
    optimizers = ["adam", "sgd", "adam_sgd_0.2", "adam_sgd_0.5", "adam_sgd_0.9"]
    tuner = tune.Tuner(partial(sam_start_points_run, base_config),
        param_space={
            "lr": tune.loguniform(1e-5, 1e-2),
            "optimizer": tune.choice(optimizers),
        }
    )
    tuner.fit()

@command
def run(config: ConfigProvider):
    setup()
    name = config.get("sweep", str, "sweep name")
    if name == "sam_start_points": sam_start_points_sweep(config)
    else: raise ValueError(f"Unknown sweep name: {name}")