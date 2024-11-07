from foundry.core.dataclasses import dataclass
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.train.reporting import Image
from foundry.util.registry import Registry

from foundry.datasets.core import Dataset
from foundry.datasets.vision import LabeledImage
from foundry.data.normalizer import Normalizer
from foundry.data import Data, PyTreeData

from pathlib import Path

import foundry.data.normalizer as normalizers
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

import itertools
import numpy as np
import optax
import wandb
import functools

import boto3
import urllib
import tempfile
import jax

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

    lr: float = 3e-4
    weight_decay: float = 1e-4

    num_visualize: int = 8

    timesteps: int = 100

    condition_type: str = "class" # one of "class" "image" or "none"
    prediction_type: str = "epsilon"

@dataclass
class ModelConfig:
    model: str
    condition_type: str
    image_shape: tuple[int]
    num_classes: int | None

    def create(self, rng : jax.Array | None = None):
        models = Registry()
        foundry.models.register_all(models)
        model = models.create(
            self.model, 
            num_classes=self.num_classes,
            out_channels=self.image_shape[-1]
        )
        if rng is None:
            return model
        pixels_sample = jnp.zeros(self.image_shape, jnp.float32)
        if self.condition_type == "class":
            cond_sample = jnp.zeros((), dtype=jnp.uint32)
        elif self.condition_type == "image":
            cond_sample = jnp.zeros(self.image_shape, jnp.float32)
        vars = F.jit(model.init)(
            rng, pixels_sample,  jnp.zeros((), dtype=jnp.uint32), 
            cond=cond_sample
        )
        return model, vars

@dataclass
class Sample:
    data: F.Array
    cond: F.Array
    label: F.Array

@dataclass
class Checkpoint:
    dataset: str
    config: ModelConfig
    schedule: DDPMSchedule
    normalizer: Normalizer
    vars: dict
    opt_state: dict
    
    def create_data(self):
        datasets = Registry()
        foundry.datasets.vision.register_all(datasets)
        dataset : Dataset[LabeledImage] = datasets.create(self.dataset)

        train_data, test_data = dataset.split("train"), dataset.split("test")

        train_data = preprocess_data(self.config.condition_type, train_data)
        test_data = preprocess_data(self.config.condition_type, test_data)
        return train_data, test_data

def create_normalizer(normalizer: str, condition_type: str, dataset: Dataset[LabeledImage]) -> Normalizer[LabeledImage]:
    image_normalizer = dataset.normalizer(normalizer).map(lambda x: x.pixels)
    orig_label_normalizer = dataset.normalizer(normalizer).map(lambda x: x.label)
    label_normalizer = orig_label_normalizer
    if condition_type == "image":
        label_normalizer = image_normalizer
    return normalizers.Compose(
        Sample(
            data=image_normalizer,
            cond=label_normalizer,
            label=orig_label_normalizer
        )
    )

def preprocess_data(condition_type: str, data : Data) -> Data[Sample]:
    if condition_type == "none":
        return data.map(
            lambda x: Sample(x.pixels, None, None)
        )
    elif condition_type == "class":
        assert hasattr(data.structure, "label")
        return data.map(
            lambda x: Sample(x.pixels, x.label, None)
        )
    elif condition_type == "image":
        # make index pairs for matching classes
        labels = data.map(lambda x: x.label).as_pytree()
        images = data.map(lambda x: x.pixels).as_pytree()
        classes = jnp.max(labels) + 1

        class_indices = [
            jnp.argwhere(labels == i)[...,0] for i in range(classes)
        ]
        per_class_max = min([len(x) for x in class_indices])
        class_indices = [x[:per_class_max] for x in class_indices]
        # a list of the pixel images for each class, balanced 
        class_images = [images[i] for i in class_indices]
        assert len(class_images) % 2 == 0
        # Contains a tuple of the image pairs
        class_pairs = tree.map(
            lambda *class_pairs: jnp.concatenate(class_pairs, axis=0),
            # pair each class with the next class, (wrapping around)
            *zip(class_images[::2], class_images[1::2]),
            *zip(class_images[1::2], class_images[2::2] + [class_images[0]])
        )
        class_labels = [jnp.full((per_class_max,), label) for label in itertools.chain(
            range(0, len(class_images), 2),
            range(1, len(class_images), 2),
        )]
        class_labels = jnp.concatenate(class_labels, axis=0)
        return PyTreeData(Sample(
            cond=class_pairs[0],
            data=class_pairs[1],
            label=class_labels,
        ))
    else:
        raise ValueError(f"Invalid condition type: {condition_type}")

def run(config: Config):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running with config: {config}")
    rng = PRNGSequence(config.seed)
    logger.info(f"Devices: {jax.devices()}")

    wandb_run = wandb.init(
        project="image-diffusion",
        config=tree.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to {wandb_run.url}")

    datasets = Registry()
    foundry.datasets.vision.register_all(datasets)
    dataset : Dataset[LabeledImage] = datasets.create(config.dataset)

    train_data, test_data = dataset.split("train"), dataset.split("test")

    train_data = preprocess_data(config.condition_type, train_data)
    test_data = preprocess_data(config.condition_type, test_data)
    normalizer = create_normalizer(config.normalizer, config.condition_type, dataset)

    augment = dataset.augmentation("generator") or (lambda _r, x: x)
    sample = normalizer.normalize(train_data[0])

    classes = len(dataset.classes) if hasattr(dataset, "classes") else None
    model_config = ModelConfig(
        model=config.model,
        condition_type=config.condition_type,
        image_shape=sample.data.shape,
        num_classes=classes
    )
    model, vars = model_config.create(next(rng))

    if config.iterations is not None:
        iterations = config.iterations
    else:
        epochs = config.epochs or 50
        iterations_per_epoch = len(train_data) // config.batch_size
        iterations = iterations_per_epoch * epochs

    logger.info("Creating model...")
    logger.info(f"Parameters: {tree.total_size(vars)}")

    @F.jit
    def diffuser(vars, cond, rng_key, noised_x, t):
        # return schedule.output_from_denoised(noised_x, t, noised_x)
        return model.apply(vars, noised_x, t - 1, cond=cond)

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
        cond_diffuser = functools.partial(diffuser, vars, sample.cond)
        loss = schedule.loss(s_rng, cond_diffuser, sample.data)
        return foundry.train.LossOutput(
            loss=loss, metrics={"loss": loss}
        )

    @F.jit
    def generate_samples(vars, labels, rng_key) -> LabeledImage:
        def do_sample(cond_unnormalized, rng_key):
            cond = normalizer.map(lambda x: x.cond).normalize(cond_unnormalized)
            denoiser = functools.partial(diffuser, vars, cond)
            pixels = schedule.sample(rng_key, denoiser, sample.data)
            pixels = normalizer.map(lambda x: x.data).unnormalize(pixels)
            return cond_unnormalized, pixels
        rngs = foundry.random.split(rng_key, config.num_visualize)
        return F.vmap(do_sample)(labels, rngs)

    batch_loss = foundry.train.batch_loss(loss)
    batch_test_loss = foundry.train.batch_loss(loss)

    train_stream = train_data.stream().shuffle(next(rng)).batch(config.batch_size)
    test_stream = test_data.stream().batch(2*config.batch_size)
    gen_stream = test_data.stream().shuffle(next(rng)).batch(config.num_visualize)
    with foundry.train.loop(train_stream, 
            iterations=iterations, rng_key=next(rng)) as loop, \
            test_stream.build() as test_stream, \
            gen_stream.build() as gen_stream:
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

                    # generate samples and log as a wandb table
                    if not gen_stream.has_next():
                        gen_stream = gen_stream.reset()
                    gen_stream, gen_batch = gen_stream.next()
                    gen_labels, gen_samples = generate_samples(vars, gen_batch.cond, sample_key)
                    gen_labels = (
                        [wandb.Image(np.array(x)) for x in gen_labels]
                        if config.condition_type == "image" else
                        [dataset.classes[i] for i in gen_labels]
                    )
                    gen_table = wandb.Table(
                        columns=["Label", "Sample"], 
                        data=[[l, wandb.Image(np.array(i))] for l, i in zip(gen_labels, gen_samples)]
                    )
                    wandb_run.log({"samples": gen_table}, step=step.iteration)

                    foundry.train.wandb.log(
                        step.iteration, test_metrics,
                        run=wandb_run, prefix="test/"
                    )
                    foundry.train.console.log(
                        step.iteration, test_metrics, prefix="test."
                    )
    checkpoint = Checkpoint(
        dataset=config.dataset,
        config=model_config,
        normalizer=normalizer,
        schedule=schedule,
        vars=vars,
        opt_state=opt_state
    )
    if config.bucket_url is not None:
        final_result_url = f"{config.bucket_url}/{wandb_run.id}/checkpoint.zarr.zip"
        foundry.util.serialize.save(final_result_url, checkpoint)
        dataset_sanitized = config.dataset.replace("/", "-")
        artifact = wandb.Artifact(f"{dataset_sanitized}-ddpm", type="model")
        artifact.add_reference(final_result_url)
        wandb_run.log_artifact(artifact)