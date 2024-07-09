from stanza.runtime import ConfigProvider, command, setup
setup()

from stanza.dataclasses import dataclass

from stanza.train import LossOutput
from stanza.random import PRNGSequence
from stanza.datasets.vision import image_class_datasets
from stanza.model import models

from functools import partial

import stanza.train
import stanza.train.wandb
import stanza.train.console
import stanza.train.sharpness

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
    download_only: bool = False

    dataset: str = "cifar/cifar10"
    normalizer: str = "standard_dev"
    model: str = "resnet/SmallResNet18"

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

    log_compiles: bool = False
    trace: bool = False

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()
        res = config.get_dataclass(defaults)
        return res

    # if we should also quantize the model

def switch_optim(
    opt_a: optax.GradientTransformation,
    opt_b: optax.GradientTransformation,
    switch_iteration: int
) -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return {"opt_a": opt_a.init(params), "opt_b": opt_b.init(params),
                "iteration": jnp.zeros((), dtype=jnp.int32)}
    def update_fn(updates, state, params=None):
        del params
        iteration = state["iteration"]
        a_updates, a_state = opt_a.update(updates, state["opt_a"])
        b_updates, b_state = opt_b.update(updates, state["opt_a"])
        updates = jax.lax.cond(switch_iteration < iteration, lambda: a_updates, lambda: b_updates)
        state = {"opt_a": a_state, "opt_b": b_state, "iteration": iteration + 1}
        return updates, state
    return optax.GradientTransformation(init_fn, update_fn)

    # if we should also quantize the model
def make_optimizer(name, lr, iterations, warmup_percent, weight_decay, sam_rho):
    schedule = optax.cosine_onecycle_schedule(
        iterations, lr or 8e-3, warmup_percent
    )
    adam_optimizer = optax.adamw(
        schedule, weight_decay=weight_decay
    )
    schedule = optax.cosine_onecycle_schedule(
        iterations, lr or 1e-2, warmup_percent
    )
    sgd_optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.sgd(schedule)
    )
    if name == "adam": optimizer = adam_optimizer
    if name == "sgd": optimizer = sgd_optimizer
    elif name == "adam_sgd_0.2":
        optimizer = switch_optim(adam_optimizer, 
            sgd_optimizer, int(0.2*iterations))
    elif name == "adam_sgd_0.5":
        optimizer = switch_optim(adam_optimizer, 
            sgd_optimizer, int(0.5*iterations))
    elif name == "adam_sgd_0.9":
        optimizer = switch_optim(adam_optimizer, 
            sgd_optimizer, int(0.9*iterations))
    if sam_rho is not None and sam_rho > 0:
        optimizer = optax.contrib.sam(
            optimizer,
            optax.sgd(sam_rho),
            opaque_mode=True,
            reset_state=False
        )
    return optimizer

def train(config: Config):
    logger.info(f"Training {config}")
    rng = PRNGSequence(config.seed)

    dataset = image_class_datasets.create(config.dataset)
    if config.download_only:
        return
    normalizer = dataset.normalizers[config.normalizer]()
    augment = (
        dataset.transforms["standard_augmentations"]()
        if "standard_augmentations" in dataset.transforms else
        lambda _r, x: x
    )

    epoch_iterations = len(dataset.splits["train"]) // config.batch_size
    iterations = config.epochs * epoch_iterations
    optimizer = make_optimizer(config.optimizer, config.lr, 
                               iterations, config.warmup_ratio,
                               config.weight_decay, config.sam_rho)
    wandb_run = wandb.init(
        project="image_classifier",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    num_classes = len(dataset.classes)
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    model = models.create(config.model, n_classes=num_classes)
    vars = model.init(next(rng), jnp.zeros_like(dataset.splits["train"].structure[0]))
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    def loss_fn(params, rng_key, sample, train=True):
        x, label = normalizer.normalize(sample)
        if train:
            x = augment(rng_key, x)
            logits_output, mutated = model.apply(params, x, 
                                                 mutable=("batch_stats",))
        else:
            logits_output = model.apply(params, x)
            mutated = {}

        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits_output, label)
        predicted_class = jnp.argmax(logits_output, axis=-1)
        accuracy = 1.*(predicted_class == label)
        return LossOutput(
            loss=ce_loss,
            metrics=dict(cross_entropy=ce_loss, accuracy=accuracy),
            var_updates=mutated
        )
    val_loss = partial(loss_fn, train=False)
    batch_loss = stanza.train.batch_loss(loss_fn)
    val_batch_loss = stanza.train.batch_loss(val_loss)
    
    vars = model.init(next(rng), jnp.zeros_like(normalizer.structure[0]))
    opt_state = optimizer.init(vars["params"])

    train_data = (
        dataset.splits["train"].stream()
            .shuffle(next(rng)).batch(config.batch_size)
    )
    sharpness_data = (
        dataset.splits["test"].stream()
            .shuffle(next(rng), resample=True).batch(config.batch_size)
    )
    test_data = dataset.splits["test"].stream().batch(2*config.batch_size)

    with stanza.train.loop(train_data, rng_key=next(rng), iterations=iterations,
                log_compiles=config.log_compiles, trace=config.trace) as loop, \
            test_data.build() as test_stream, \
            sharpness_data.build() as sharpness_stream:

        for epoch in loop.epochs():
            for step in epoch.steps():
                opt_state, vars, metrics = stanza.train.step(
                    batch_loss, optimizer,
                    opt_state, vars,
                    step.rng_key, step.batch 
                )
                stanza.train.wandb.log(
                    step.iteration, metrics,
                    run=wandb_run, prefix="train/"
                )
                # print to the console every 100 iterations
                if step.iteration % 100 == 0:
                    stanza.train.console.log(
                        step.iteration, metrics, prefix="train."
                    )
                # validate + log every 500 steps
                if step.iteration % 100 == 0:
                    test_stream, test_metrics = stanza.train.eval_stream(
                        val_batch_loss, vars, next(rng), test_stream
                    )
                    stanza.train.console.log(
                        step.iteration, test_metrics,
                        prefix="test."
                    )
                    stanza.train.wandb.log(
                        step.iteration, test_metrics,
                        prefix="test/", run=wandb_run
                    )
                if step.iteration % 200 == 0:
                    sharpness_stream, sharpness_batch = sharpness_stream.next()
                    sharpness_metrics = stanza.train.sharpness.sharpness_stats(
                        val_loss, vars, next(rng), sharpness_batch,
                        batch_size=max(64, config.batch_size)
                    )
                    stanza.train.console.log(
                        step.iteration, sharpness_metrics,
                        prefix="test."
                    )
                    stanza.train.wandb.log(
                        step.iteration, sharpness_metrics,
                        prefix="test/", run=wandb_run
                    )

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    train(Config.parse(config))