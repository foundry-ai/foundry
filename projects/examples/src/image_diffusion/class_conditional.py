from stanza import dataclasses, partial
from stanza.struct.args import command

from stanza.random import PRNGSequence
from stanza.datasets import image_class_datasets
from stanza.diffusion import DDPMSchedule

from stanza.nn.models.unet import DiffusionUNet

import stanza.util
import stanza.train as st
import stanza.train.wandb

from common import TrainConfig, OptimizerConfig, AdamConfig

import stanza.graphics

import jax
import jax.numpy as jnp

import wandb

import logging
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Config:
    seed: int = 42
    diffusion_steps: int = 50
    dataset: str = "mnist"
    normalizer: str = "hypercube"
    train: TrainConfig = TrainConfig(
        optimizer=AdamConfig(3e-4, "cosine", "linear", 100)
    )

def train(config: Config):
    rng = PRNGSequence(config.seed)
    dataset = image_class_datasets.create(config.dataset)
    train_data, test_data = dataset.splits["train"], dataset.splits["test"]

    wandb_run = wandb.init(
        project="image_diffusion",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    normalizer = dataset.normalizers[config.normalizer]
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps
    )
    model = DiffusionUNet(base_channels=32, time_embed_dim=64)

    vars = jax.jit(model.init)(next(rng),
            jnp.zeros_like(normalizer.structure[0]), 0.)

    @jax.jit
    def generate_samples(params, rng_key):
        @jax.jit
        def sample(vars, rng_key):
            denoiser = lambda _, x, t: model.apply(vars, x, t - 1)
            return normalizer.unnormalize(schedule.sample(rng_key, denoiser, normalizer.structure))
        sample = jax.vmap(sample, in_axes=(None, 0))
        samples = sample(params, jax.random.split(rng_key, 64))
        return stanza.graphics.image_grid(samples)

    def generate_hook(rng, train_state):
        return {
            "samples": st.Image(generate_samples(train_state.vars, next(rng)))
        }

    def loss_fn(params, _iteration, rng_key, sample):
        image, label = normalizer.normalize(sample)
        denoiser = lambda _, x, t: model.apply(params, x, t - 1)
        loss = schedule.loss(rng_key, denoiser, image)
        return st.LossOutput(
            loss=loss,
            metrics={"loss": loss}
        )

    batch_loss_fn = st.batch_loss(loss_fn)

    vars = config.train.fit(
        data=train_data,
        batch_loss_fn=batch_loss_fn,
        rng_key=next(rng),
        init_vars=vars,
        donate_init_vars=True,
        hooks=[
            st.every_n_iterations(100,
                st.wandb.wandb_logger(run=wandb_run, metrics=True),
                st.console_logger(metrics=True)
            ),
            st.every_n_iterations(2000,
                st.wandb.wandb_logger(
                    generate_hook, run=wandb_run
                )
            )
        ]
    )
    return vars

# the cli main
@command(Config)
def run(config: Config):
    logger.setLevel(logging.DEBUG)
    train(config)