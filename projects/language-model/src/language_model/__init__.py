from stanza.runtime import ConfigProvider, command, setup
setup()

from stanza.dataclasses import dataclass

from stanza.train import LossOutput
from stanza.random import PRNGSequence
from stanza.datasets.nlp import datasets
from stanza.model import models

from functools import partial

import stanza.train
import stanza.train.wandb
import stanza.train.console

import flax
import rich
import wandb

import optax
import jax
import jax.numpy as jnp

import stanza.util
import logging
logger = logging.getLogger(__name__)


@dataclass
class Config:
    seed: int = 42
    dataset: str = "tinystories"
    model: str = "gpt2/nano"

    epochs: int = 50
    warmup_ratio: float = 0.01
    batch_size: int = 128
    optimizer: str = "adam"
    lr: float | None = None
    weight_decay: float = 1e-4

    # sam-related parameters
    sam_rho: float = 0. # None if SAM is disabled
    sam_start: float = 0. # percentage of training through which to start sam
    sam_percent: float = 1. # percentage of training to use sam, if enabled

    # sharpness measure
    sharpness: bool = False

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()
        res = config.get_dataclass(defaults)
        return res

    # if we should also quantize the model
def make_optimizer(name, lr, iterations, warmup_percent, weight_decay):
    lr = lr or (5e-3 if name == "adam" else 2e-1)
    schedule = optax.cosine_onecycle_schedule(
        iterations, lr, warmup_percent
    )
    if name == "adam":
        return optax.adamw(
            schedule, weight_decay=weight_decay
        )
    elif name == "sgd":
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(schedule)
        )

def train(config: Config):
    logger.info(f"Training {config}")
    rng = PRNGSequence(config.seed)
    dataset = datasets.create(config.dataset)

    iterations_per_epoch = len(dataset.splits["train"]) // config.batch_size
    iterations = config.epochs * iterations_per_epoch
    optimizer = make_optimizer(config.optimizer, config.lr, 
                               iterations, config.warmup_ratio,
                               config.weight_decay)
    if config.sam_rho is not None and config.sam_rho > 0:
        optimizer = optax.contrib.sam(
            optimizer,
            optax.sgd(config.sam_rho),
            opaque_mode=True,
            reset_state=False
        )
    wandb_run = wandb.init(
        project="image_classifier",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")
    model = models.create(config.model, vocab_size=dataset.tokenizer.vocab_size)
    vars = jax.jit(partial(model.init, deterministic=True))(
        next(rng),
        jnp.zeros_like(dataset.splits["train"].structure), 
    )
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    def loss_fn(params, rng_key, sample, train=True):
        pass
    
    batch_loss = stanza.train.batch_loss(loss_fn)
    val_batch_loss = stanza.train.batch_loss(
        partial(loss_fn, train=False)
    )
    opt_state = optimizer.init(vars["params"])

    # load all the test data directly into memory
    test_data = dataset.splits["test"].as_pytree()
    with stanza.train.loop(dataset.splits["train"], 
                rng_key=next(rng),
                iterations=iterations,
                batch_size=config.batch_size,
                resample=True,
                progress=True) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                print(step.batch)
                opt_state, vars, metrics = stanza.train.step(
                    batch_loss, optimizer,
                    opt_state, vars,
                    step.rng_key, step.batch 
                )
                print("stepped")
                stanza.train.wandb.log(
                    step.iteration, metrics,
                    run=wandb_run
                )
                # print to the console every 100 iterations
                if step.iteration % 100 == 0:
                    stanza.train.console.log(
                        step.iteration, metrics
                    )
                # validate + log every 500 steps
                if step.iteration % 500 == 0:
                    sharpness_metrics = None
                    test_metrics = stanza.train.eval(
                        val_batch_loss, vars, next(rng), test_data
                    )
                    stanza.train.console.log(
                        step.iteration, 
                        sharpness_metrics, test_metrics,
                        prefix="test"
                    )
                    stanza.train.wandb.log(
                        step.iteration,
                        sharpness_metrics, test_metrics,
                        prefix="test", run=wandb_run
                    )

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    train(Config.parse(config))