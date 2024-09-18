from foundry.core.dataclasses import dataclass
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.train.reporting import Image
from foundry.datasets.vision import LabeledImage
from foundry.util.registry import Registry

from pathlib import Path

import foundry.graphics
import foundry.train
import foundry.train.wandb
import foundry.train.console
import foundry.random
import foundry.util.serialize
import foundry.datasets.vision
import foundry.models

import foundry.numpy as jnp
import foundry.core as F
import optax
import wandb
import functools

import boto3
import urllib
import tempfile

import logging
logger = logging.getLogger("image_diffusion")

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 128
    model: str = "diffusion/unet/small"

    dataset: str = "cifar10"
    normalizer: str = "hypercube"

    bucket_url: str = "s3://wandb-data"

    epochs: int | None = None
    iterations: int | None = None

    use_estimator: bool = False

    lr: float = 3e-4
    weight_decay: float = 1e-4

    num_visualize: int = 8

    timesteps: int = 100
    prediction_type: str = "epsilon"

@dataclass
class ModelConfig:
    model: str

@dataclass
class Checkpoint:
    config: ModelConfig
    vars: dict
    opt_state: dict

def run(config: Config):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running with config: {config}")
    rng = PRNGSequence(config.seed)

    wandb_run = wandb.init(
        project="image-diffusion",
        config=tree.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to {wandb_run.url}")

    datasets = Registry()
    models = Registry()
    foundry.datasets.vision.register_all(datasets)
    foundry.models.register_all(models)

    dataset = datasets.create(config.dataset)
    train_data, test_data = dataset.split("train"), dataset.split("test")
    normalizer = dataset.normalizer(config.normalizer)
    augment = dataset.augmentation("generator") or (lambda _r, x: x)
    sample = normalizer.normalize(train_data[0])

    if config.iterations is not None:
        iterations = config.iterations
    else:
        epochs = config.epochs or 50
        iterations_per_epoch = len(train_data) // config.batch_size
        iterations = iterations_per_epoch * epochs

    logger.info("Creating model...")
    model = models.create(
        config.model, 
        num_classes=len(dataset.classes)
    )
    vars = F.jit(model.init)(next(rng), sample.pixels, 0, cond=None)
    logger.info(f"Parameters: {tree.total_size(vars)}")

    @F.jit
    def diffuser(vars, cond, rng_key, noised_x, t):
        # return schedule.output_from_denoised(noised_x, t, noised_x)
        return model.apply(vars, noised_x, t - 1, cond=None)

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps, prediction_type=config.prediction_type,
        clip_sample_range=2.
    )

    lr_schedule = optax.cosine_onecycle_schedule(
        iterations, config.lr, pct_start=0.05
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    # opt_state = optimizer.init(vars["params"])
    # lr_schedule = optax.cosine_onecycle_schedule(iterations, 1e-2, pct_start=0.05)
    # optimizer = optax.sgd(lr_schedule)
    opt_state = optimizer.init(vars["params"])

    @F.jit
    def loss(vars, rng_key, sample):
        s_rng, a_rng = foundry.random.split(rng_key)
        sample = normalizer.normalize(sample)
        label = sample.label
        pixels = augment(a_rng, sample.pixels)
        cond_diffuser = functools.partial(diffuser, vars, label)
        loss = schedule.loss(s_rng, cond_diffuser, pixels)
        return foundry.train.LossOutput(
            loss=loss, metrics={"loss": loss}
        )

    @F.jit
    def test_loss(vars, rng_key, sample):
        sample = normalizer.normalize(sample)
        label = sample.label
        cond_diffuser = functools.partial(diffuser, vars, label)
        Ts = range(1, config.timesteps + 1, config.timesteps // 10)
        rngs = foundry.random.split(rng_key, len(Ts))
        losses = F.vmap(schedule.loss, in_axes=(0, None, None, 0))(rngs, cond_diffuser, sample.pixels, jnp.array(Ts))
        metrics = {f"loss_t_{t}": l for t, l in zip(Ts, losses)}
        loss = jnp.mean(losses)
        return foundry.train.LossOutput(
            loss, metrics=metrics
        )

    @F.jit
    def generate_samples(vars, rng_key) -> LabeledImage:
        def do_sample(rng_key):
            l_rng, s_rng = foundry.random.split(rng_key)
            label = foundry.random.randint(l_rng, (), 0, len(dataset.classes))
            denoiser = functools.partial(diffuser, vars, label)
            pixels = schedule.sample(s_rng, denoiser, sample.pixels)
            return normalizer.unnormalize(LabeledImage(pixels, label))
        rngs = foundry.random.split(rng_key, config.num_visualize)
        samples = F.vmap(do_sample)(rngs)
        return foundry.graphics.image_grid(samples.pixels)

    batch_loss = foundry.train.batch_loss(loss)
    batch_test_loss = foundry.train.batch_loss(test_loss)

    train_stream = train_data.stream().shuffle(next(rng)).batch(config.batch_size)
    test_stream = test_data.stream().batch(2*config.batch_size)
    with foundry.train.loop(train_stream, 
            iterations=iterations, rng_key=next(rng)) as loop, \
            test_stream.build() as test_stream:
        for epoch in loop.epochs():
            for step in epoch.steps():
                train_key, test_key = foundry.random.split(step.rng_key)
                opt_state, vars, grad_norm, metrics = foundry.train.step(
                    batch_loss, optimizer,
                    opt_state, vars,
                    train_key, step.batch,
                    return_grad_norm=True
                )
                foundry.train.wandb.log(
                    step.iteration, metrics, {"lr": lr_schedule(step.iteration), "grad_norm": grad_norm},
                    run=wandb_run, prefix="train/"
                )
                if step.iteration % 100 == 0:
                    foundry.train.console.log(
                        step.iteration, metrics, {"lr": lr_schedule(step.iteration)},
                        prefix="train."
                    )
                if step.iteration % 500 == 0:
                    eval_key, sample_key = foundry.random.split(test_key)
                    test_stream, test_metrics = foundry.train.eval_stream(
                        batch_test_loss, vars, eval_key, test_stream
                    )
                    image = generate_samples(vars, sample_key)
                    foundry.train.wandb.log(
                        step.iteration, test_metrics, 
                        run=wandb_run, prefix="test/"
                    )
                    foundry.train.wandb.log(
                        step.iteration,
                        {"images": Image(image)},
                        run=wandb_run, prefix="generated/"
                    )
                    foundry.train.console.log(
                        step.iteration, test_metrics, prefix="test."
                    )
    checkpoint = Checkpoint(
        config=ModelConfig(model=config.model),
        vars=vars,
        opt_state=opt_state
    )
    if config.bucket_url is not None:
        final_result_url = f"{config.bucket_url}/{wandb_run.id}/checkpoint.zarr.zip"
        foundry.util.serialize.save(final_result_url, checkpoint)
        artifact = wandb.Artifact(f"{config.dataset.replace('/', '-')}-ddpm", type="model")
        artifact.add_reference(final_result_url)
        wandb_run.log_artifact(artifact)