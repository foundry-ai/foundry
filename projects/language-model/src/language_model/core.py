from stanza.runtime import ConfigProvider
from stanza.dataclasses import dataclass
from stanza.train import LossOutput
from stanza.random import PRNGSequence
from stanza.datasets.nlp import datasets
from stanza.model import models

from functools import partial

import stanza.train
import stanza.train.ray
import stanza.train.wandb
import stanza.train.console
import stanza.train.sharpness

import flax
import rich

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

    download_only: bool = False

    iterations: int = 100
    warmup_ratio: float = 0.01
    batch_size: int = 4
    optimizer: str = "adam"
    lr: float | None = None
    weight_decay: float = 1e-4

    # sam-related parameters
    sam_rho: float = 0. # None if SAM is disabled
    sam_start: float = 0. # percentage of training through which to start sam

    log_compiles: bool = False
    trace: bool = False

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()
        res = config.get_dataclass(defaults)
        return res

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

def train(wandb_run, config: Config):
    logger.info(f"Training {config}")
    rng = PRNGSequence(config.seed)
    dataset = datasets.create(config.dataset, download_only=config.download_only)
    if config.download_only:
        return

    iterations = config.iterations
    optimizer = make_optimizer(config.optimizer, config.lr, 
                               iterations, config.warmup_ratio,
                               config.weight_decay, config.sam_rho)
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")
    model = models.create(config.model, vocab_size=dataset.tokenizer.vocab_size)
    vars = jax.jit(partial(model.init, deterministic=True))(
        next(rng),
        jnp.zeros_like(dataset.splits["train"].structure), 
    )
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")
    logger.info(f"Vocabulary size: [blue]{dataset.tokenizer.vocab_size}[/blue]")

    def loss_fn(params, rng_key, tokens, train=True):
        X, Y = tokens[:-1], tokens[1:]
        logits = model.apply(
            params, X,
            not train, 
            rngs={'dropout': rng_key}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
        stats = {"loss": loss}
        return LossOutput(loss=loss, metrics=stats)

    val_loss_fn = partial(loss_fn, train=False)
    batch_loss = stanza.train.batch_loss(loss_fn)
    val_batch_loss = stanza.train.batch_loss(val_loss_fn)
    opt_state = jax.jit(optimizer.init)(vars["params"])
    logger.info("Initialized optimizer.")

    train_data = (
        dataset.splits["train"].stream()
            .shuffle(next(rng)).batch(config.batch_size)
    )
    sharpness_data = (
        dataset.splits["test"].stream()
            .shuffle(next(rng), resample=True).batch(config.batch_size)
    )
    test_data = dataset.splits["test"].stream().batch(config.batch_size)

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
                        step.iteration, metrics
                    )
                    test_stream, test_metrics = stanza.train.eval_stream(
                        val_batch_loss, vars, next(rng),
                        test_stream, batches=16
                    )
                    sharpness_stream, sharpness_batch = sharpness_stream.next()
                    sharpness_metrics = stanza.train.sharpness.sharpness_stats(
                        val_loss_fn, vars, next(rng), sharpness_batch,
                        batch_size=max(64, config.batch_size)
                    )
                    stanza.train.console.log(
                        step.iteration, 
                        sharpness_metrics, test_metrics, prefix="test."
                    )
                    stanza.train.ray.report(step.iteration, test_metrics)
                    stanza.train.wandb.log(
                        step.iteration,
                        sharpness_metrics, test_metrics,
                        prefix="test/", run=wandb_run
                    )