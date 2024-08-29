from foundry.runtime import ConfigProvider, command, setup
from .core import Config, train

import logging
import foundry.util

logger = logging.getLogger(__name__)

@command
def run(config: ConfigProvider):
    setup()
    import wandb
    logger.setLevel(logging.DEBUG)
    config = Config.parse(config)
    wandb_run = wandb.init(
        project="language_model",
        config=foundry.util.flatten_to_dict(config)[0]
    )
    train(wandb_run, config)