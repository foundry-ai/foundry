from stanza.struct.args import command
from stanza import struct

import stanza.train as st
import stanza.train.wandb as stw

import stanza.train.wandb
import stanza.util.summary

from stanza.util import lanczos

from stanza.random import PRNGSequence
from stanza.datasets import image_class_datasets

from functools import partial
from common import TrainConfig, AdamConfig, SAMConfig, SGDConfig

import net

import flax
import rich
import wandb

import optax
import jax
import jax.numpy as jnp

import stanza.util
import logging
logger = logging.getLogger(__name__)


@struct.dataclass
class Config:
    seed: int = 42
    summary: bool = False
    dataset: str = "cifar10"
    normalizer: str = "standard_dev"
    model: str = "SmallResNet18"

    lr: float | None = None
    lr_schedule: str = "cosine"
    cycles: int = 1 # cycles of the lr schedule to play
    cycle_mult : float = 2.
    epochs: int = 50
    batch_size: int = 128
    optimizer: str = "adam"

    # sam-related parameters
    sam_rho: float | None = None # None if SAM is disabled
    sam_start: float = 0. # percentage of training through which to start sam
    sam_percent: float = 1. # percentage of training to use sam, if enabled

    # sharpness measure
    sharpness: bool = False

    # if we should also quantize the model

def train(config: Config):
    rng = PRNGSequence(config.seed)

    if config.optimizer == "adam":
        optimizer = AdamConfig(config.lr or 5e-3,
            config.lr_schedule,
            cycles=config.cycles,
            cycle_mult=config.cycle_mult,
            weight_decay=1e-4
        )
    elif config.optimizer == "sgd":
        optimizer = SGDConfig(config.lr or 1e-1,
            config.lr_schedule, weight_decay=1e-4,
            cycles=config.cycles,
            cycle_mult=config.cycle_mult
        )
    wandb_run = wandb.init(
        project="image_classifier",
        config=stanza.util.flatten_to_dict(config)[0]
    )

    if config.sam_rho is not None and config.sam_rho > 0:
        optimizer = SAMConfig(
            forward=optimizer,
            backward=SGDConfig(config.sam_rho),
            start_percent=config.sam_start,
            run_percent=config.sam_percent
        )
    elif config.sam_start > 0 or config.sam_percent < 1:
        logger.error("SAM is disabled but SAM parameters are set")
        wandb_run.finish()
        return

    train_config = TrainConfig(
        batch_size=config.batch_size,
        epochs=config.epochs,
        optimizer=optimizer
    )
    dataset = image_class_datasets.create(config.dataset)
    num_classes = len(dataset.classes)
    normalizer = dataset.normalizers[config.normalizer]()
    augment = (
        dataset.transforms["standard_augmentations"]()
        if "standard_augmentations" in dataset.transforms else
        lambda _r, x: x
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    Model = getattr(net, config.model)
    model = Model(n_classes=num_classes)
    vars = jax.jit(model.init)(next(rng), jnp.zeros_like(normalizer.structure[0]))
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")
    
    if config.summary:
        rich.print(
            stanza.util.summary.tabulate(
                model, 
                jax.random.key(0), 
                compute_flops=False
            )(jnp.zeros_like(normalizer.structure[0]))
        )

    def loss_fn(params, _iteration, rng_key, sample, train=True):
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
        return st.LossOutput(
            loss=ce_loss,
            metrics=dict(cross_entropy=ce_loss, accuracy=accuracy),
            var_updates=mutated
        )
    
    batch_loss = st.batch_loss(loss_fn)
    val_batch_loss = st.batch_loss(partial(loss_fn, train=False))

    def sharpness_stats(vars, _iteration, rng_key, batch):
        batch = jax.vmap(normalizer.normalize)(batch)
        # only compute the sharpenss wrt trainable params
        other_vars, params = flax.core.pop(vars, "params")
        if config.sharpness:
            def loss(params, sample):
                vars = {"params": params, **other_vars}
                x, label = sample
                logits_out = model.apply(vars, x)
                return optax.softmax_cross_entropy_with_integer_labels(logits_out, label)
            hvp_at = partial(lanczos.net_batch_hvp, 
                loss, batch, config.batch_size
            )
            sharpness_stats = lanczos.net_sharpness_statistics(rng_key, hvp_at, params)
        else:
            sharpness_stats = {}
        return st.LossOutput(
            metrics=sharpness_stats
        )


    vars = train_config.fit(
        data=dataset.splits["train"],
        batch_loss_fn=batch_loss,
        rng_key=next(rng),
        init_vars=vars,
        donate_init_vars=True,
        hooks=[
            stw.wandb_logger(run=wandb_run, metrics=True, prefix="train/"),
            st.every_n_iterations(500,
                st.console_logger(metrics=True, prefix="train.")
            ),
            st.every_n_iterations(500,
                st.validate(
                    data=dataset.splits["train"],
                    batch_loss_fn=sharpness_stats,
                    batch_size=8*config.batch_size,
                    batches=1, # run on a single batch, 8*the regular size
                               # we will scan over the 8 batches to sum up the hvp
                    log_hooks=[
                        st.wandb.wandb_logger(run=wandb_run, prefix="sharpness/"),
                        st.console_logger(prefix="sharpness.")
                    ]
                ),
            ),
            st.every_n_iterations(500,
                st.validate(
                    data=dataset.splits["test"],
                    batch_loss_fn=val_batch_loss,
                    batch_size=None,
                    log_hooks=[
                        st.wandb.wandb_logger(run=wandb_run, prefix="test/"),
                        st.console_logger(prefix="test.")
                    ]
                ),
            ),
        ]
    )

@command(Config)
def run(config: Config):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Training {config}")
    train(config)