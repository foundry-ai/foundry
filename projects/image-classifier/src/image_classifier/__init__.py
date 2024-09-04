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

import os
import wandb
import math
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

    epochs: int | None = None
    iterations: int | None = None

    warmup_ratio: float = 0.01

    schedule : str = "exponential"
    schedule_decay : float = 0.01

    batch_size: int = 128
    optimizer: str = "adam"
    lr: float | None = None
    weight_decay: float = 1e-5

    # sam-related parameters
    sam_rho: float = 0. # None if SAM is disabled
    sam_start: float = 0. # percentage of training through which to start sam

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
        def do_a():
            a_updates, a_state = opt_a.update(updates, state["opt_a"], params, **extra_args)
            return a_updates, a_state, state["opt_b"]
        def do_b():
            b_updates, b_state = opt_b.update(updates, state["opt_b"], params, **extra_args)
            return b_updates, state["opt_a"], b_state
        updates, a_state, b_state = jax.lax.cond(iteration < switch_iteration, do_a, do_b)
        state = {"opt_a": a_state, "opt_b": b_state, "iteration": iteration + 1}
        return updates, state
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def make_optimizer(name, lr, iterations, warmup_percent, 
                    schedule, schedule_decay,
                    weight_decay, sam_rho, sam_start):
    def make_schedule(lr):
        warmup_steps = int(warmup_percent * iterations)
        if schedule == "constant":
            return optax.constant_schedule(lr)
        elif schedule == "linear":
            return optax.linear_schedule(0, lr, warmup_steps)
        elif schedule == "exponential":
            decay_rate = 0.7
            # don't decay to less than 1e-4 of the original rate
            max_schedule_decay = max(1e-4, schedule_decay)
            num_decays = math.ceil(math.log(max_schedule_decay)/math.log(decay_rate))
            return optax.warmup_exponential_decay_schedule(
                 lr*0.01, lr, warmup_steps,
                 staircase=True,
                 transition_steps=(iterations - warmup_steps) // num_decays,
                 decay_rate=decay_rate,
                 end_value=lr*schedule_decay
            )
        elif schedule == "cosine":
            return optax.warmup_cosine_decay_schedule(
                lr*0.01, lr, warmup_steps, iterations,
                schedule_decay * lr
            )
        else:
            raise RuntimeError(f"Unknown schedule {schedule}")

    adam_schedule = make_schedule(lr or 5e-4)
    sgd_schedule = make_schedule(lr or 1e-2)

    optimizers = {
        "adam": (adam_schedule, optax.adamw(adam_schedule, weight_decay=weight_decay)),
        "sgd": (sgd_schedule, optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(sgd_schedule)
        ))
    }
    for opt_a in ["adam", "sgd"]:
        for opt_b in ["adam", "sgd"]:
            if opt_a == opt_b: continue
            for percent in [0.2, 0.5, 0.9]:
                switch_iteration = int(percent*iterations)
                a_sched, a_opt = optimizers[opt_a]
                b_sched, b_opt = optimizers[opt_b]
                switch_schedule = lambda i: a_sched(i) if i < int(percent*iterations) else b_sched(i)
                optimizers[f"{opt_a}_{opt_b}_{percent}"] = (
                    switch_schedule, switch_optim(a_opt, b_opt, switch_iteration)
                )
    if name not in optimizers:
        raise RuntimeError(f"Unknown optimizer {name}")
    schedule, optimizer = optimizers[name]

    if sam_rho is not None and sam_rho > 0:
        sam_optimizer = optax.contrib.sam(
            optimizer,
            optax.sgd(sam_rho),
            opaque_mode=True,
            reset_state=True
        )
        if sam_start > 0: # switch sam on later
            optimizer = switch_optim(optimizer, sam_optimizer, int(sam_start*iterations))
        else:
            optimizer = sam_optimizer
    return optimizer, schedule

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

    if config.iterations is not None:
        iterations = config.iterations
    else:
        epochs = config.epochs or 50
        epoch_iterations = len(dataset.splits["train"]) // config.batch_size
        iterations = epochs * epoch_iterations

    optimizer, schedule = make_optimizer(config.optimizer, config.lr, 
                               iterations, config.warmup_ratio,
                               config.schedule, config.schedule_decay,
                               config.weight_decay, config.sam_rho, config.sam_start)
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
                    step.iteration, metrics, {"lr": schedule(step.iteration)},
                    run=wandb_run, prefix="train/"
                )
                if step.iteration % 100 == 0:
                    foundry.train.console.log(
                        step.iteration, metrics, {"lr": schedule(step.iteration)},
                        prefix="train."
                    )
                # validate + log every 500 steps
                if step.iteration % 300 == 0:
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
                if step.iteration % 600 == 0:
                    sharpness_stream, sharpness_batch = sharpness_stream.next()
                    sharpness_metrics = foundry.train.sharpness.sharpness_stats(
                        val_loss, vars, next(rng), sharpness_batch,
                        batch_size=max(64, config.batch_size)
                    )
                    foundry.train.console.log(
                        step.iteration, sharpness_metrics,
                        prefix="sharpness."
                    )
                    foundry.train.wandb.log(
                        step.iteration, sharpness_metrics,
                        prefix="sharpness/", run=wandb_run
                    )