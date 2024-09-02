from foundry.core.dataclasses import dataclass

from foundry.core import tree
from foundry.train import LossOutput
from foundry.random import PRNGSequence
from foundry.datasets.vision import image_class_datasets
from foundry.models import models

from functools import partial

import foundry.train
import foundry.train.wandb
import foundry.train.console
import foundry.train.sharpness

import wandb
import optax
import jax
import foundry.numpy as jnp

import foundry.util
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

def switch_optim(
    opt_a: optax.GradientTransformation,
    opt_b: optax.GradientTransformation,
    switch_iteration: int
) -> optax.GradientTransformation:
    def init_fn(params):
        new_params = {"opt_a": opt_a.init(params), "opt_b": opt_b.init(params),
                "iteration": jnp.zeros((), dtype=jnp.int32)}
        return new_params
    def update_fn(updates, state, params=None, **extra_args):
        iteration = state["iteration"]
        a_updates, a_state = opt_a.update(updates, state["opt_a"], params)
        b_updates, b_state = opt_b.update(updates, state["opt_a"], params)
        updates = jax.lax.cond(switch_iteration < iteration, lambda: a_updates, lambda: b_updates)
        state = {"opt_a": a_state, "opt_b": b_state, "iteration": iteration + 1}
        return updates, state
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

    # if we should also quantize the model
def make_optimizer(name, lr, iterations, warmup_percent, weight_decay, sam_rho):
    schedule = optax.cosine_onecycle_schedule(
        iterations, lr or 5e-3, warmup_percent
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
    elif name == "sgd_adam_0.2":
        optimizer = switch_optim(sgd_optimizer, 
            adam_optimizer, int(0.2*iterations))
    elif name == "sgd_adam_0.5":
        optimizer = switch_optim(sgd_optimizer, 
            adam_optimizer, int(0.2*iterations))
    elif name == "sgd_adam_0.9":
        optimizer = switch_optim(sgd_optimizer, 
            adam_optimizer, int(0.9*iterations))
    if sam_rho is not None and sam_rho > 0:
        optimizer = optax.contrib.sam(
            optimizer,
            optax.sgd(sam_rho),
            opaque_mode=True,
            reset_state=False
        )
    return optimizer

def run(config: Config):
    logger.setLevel(logging.DEBUG)
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
        config=tree.flatten_to_dict(config)[0]
    )
    num_classes = len(dataset.classes)
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    model = models.create(config.model, n_classes=num_classes)
    vars = model.init(next(rng), jnp.zeros_like(dataset.splits["train"].structure.pixels))
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    def loss_fn(params, rng_key, sample, train=True):
        normalzied = normalizer.normalize(sample)
        x, label = normalzied.pixels, normalzied.label
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
    batch_loss = foundry.train.batch_loss(loss_fn)
    val_batch_loss = foundry.train.batch_loss(val_loss)
    
    vars = model.init(next(rng), jnp.zeros_like(normalizer.structure.pixels))
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
    with foundry.train.loop(train_data, rng_key=next(rng), iterations=iterations,
                log_compiles=config.log_compiles, trace=config.trace) as loop, \
            test_data.build() as test_stream, \
            sharpness_data.build() as sharpness_stream:

        for epoch in loop.epochs():
            for step in epoch.steps():
                opt_state, vars, metrics = foundry.train.step(
                    batch_loss, optimizer,
                    opt_state, vars,
                    step.rng_key, step.batch 
                )
                foundry.train.wandb.log(
                    step.iteration, metrics,
                    run=wandb_run, prefix="train/"
                )
                # print to the console every 100 iterations
                if step.iteration % 100 == 0:
                    foundry.train.console.log(
                        step.iteration, metrics, prefix="train."
                    )
                # validate + log every 500 steps
                if step.iteration % int(128*200 / config.batch_size) == 0:
                    test_stream, test_metrics = foundry.train.eval_stream(
                        val_batch_loss, vars, next(rng), test_stream
                    )
                    foundry.train.console.log(
                        step.iteration, test_metrics,
                        prefix="test."
                    )
                    foundry.train.wandb.log(
                        step.iteration, test_metrics,
                        prefix="test/", run=wandb_run
                    )
                if step.iteration % int(128*500 / config.batch_size) == 0:
                    sharpness_stream, sharpness_batch = sharpness_stream.next()
                    sharpness_metrics = foundry.train.sharpness.sharpness_stats(
                        val_loss, vars, next(rng), sharpness_batch,
                        batch_size=max(64, config.batch_size)
                    )
                    foundry.train.console.log(
                        step.iteration, sharpness_metrics,
                        prefix="test."
                    )
                    foundry.train.wandb.log(
                        step.iteration, sharpness_metrics,
                        prefix="test/", run=wandb_run
                    )