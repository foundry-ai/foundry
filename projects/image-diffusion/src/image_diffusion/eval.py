from foundry.core.dataclasses import dataclass, replace
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.data import PyTreeData

import foundry.train
from foundry.train import LossOutput

from typing import Any
from collections.abc import Sequence

import foundry.core as F

from pathlib import Path
from rich.progress import track

import foundry.util.serialize
import foundry.random
import foundry.numpy as npx

import jax
import flax.linen as nn
import optax

import wandb
import tempfile
import logging

logger = logging.getLogger("eval")


@dataclass
class Config:
    run : str = "dpfrommer-projects/image-diffusion/runs/1vpg8vbk"
    bucket_url: str = "s3://wandb-data"
    seed: int = 42

    t_pct: float = 0.5

    samples_per_cond: int = 128
    eval_per_cond: int = 4
    batch_size: int = 64
    batches: int = 64

    step_interval: int = 10_000
    only_final: bool = False

@dataclass
class EvaluationInputs:
    vars: Any
    model_apply: Any
    schedule: DDPMSchedule
    keypoints: Any

    sample_shape: Sequence[int]
    samples_per_cond: int
    eval_per_cond: int
    batch_size: int
    batches: int
    t_pct: float

@dataclass
class EvaluationOutputs:
    cond: jax.Array

    # These have an extra batch array
    # per-cond
    ts: jax.Array
    nw_error: jax.Array
    lin_error: jax.Array = None

    # variables for the alpha-prediction network
    alpha_vars: Any = None
    keypoints: Any = None

    # the original checkpoint_uri
    checkpoint_uri: str | None = None

@dataclass
class GenerationData:
    cond: jax.Array
    reverse_sample: jax.Array
    t: jax.Array
    out_keypoints: jax.Array
    out_model: jax.Array

@F.jit
def evaluate_cond(inputs: EvaluationInputs, cond : Any, rng_key : jax.Array):
    schedule = inputs.schedule

    samples_per_cond = max(inputs.samples_per_cond, inputs.eval_per_cond)
    eval_per_cond = inputs.eval_per_cond

    def sample(rng_key):
        r_rng, s_rng = foundry.random.split(rng_key)
        denoiser = lambda rng_key, x, t: inputs.model_apply(inputs.vars, x, t-1, cond=cond)
        sample, trajectory = schedule.sample(
            rng_key, denoiser, npx.zeros(inputs.sample_shape),
            trajectory=True
        )
        t = int(schedule.num_steps * inputs.t_pct) # foundry.random.randint(s_rng, (), minval=1, maxval=trajectory.shape[0])
        return sample, trajectory[t], t

    rng_keys = foundry.random.split(rng_key, samples_per_cond)
    samples, reverse_samples, ts = jax.lax.map(
        sample, rng_keys, 
        batch_size=8
    )
    samples = samples[eval_per_cond:]
    reverse_samples = reverse_samples[:eval_per_cond]
    ts = ts[:eval_per_cond]

    def eval(eval_inputs):
        reverse_sample, t = eval_inputs
        model = lambda cond: inputs.model_apply(
            inputs.vars, reverse_sample, t-1, 
            cond=cond
        )
        keypoints_out = jax.lax.map(model, inputs.keypoints)
        model_out = model(cond)
        nw_out = schedule.output_from_denoised(
            reverse_sample, t,
            schedule.compute_denoised(
                reverse_sample, t, samples
            )
        )
        nw_err = npx.linalg.norm(nw_out - model_out)
        return nw_err, GenerationData(
            cond, reverse_sample, t, keypoints_out, model_out
        )
    nw_errs, gen_data = jax.lax.map(eval, (reverse_samples, ts))
    return EvaluationOutputs(
        cond, ts, nw_errs
    ), gen_data

def evaluate_checkpoint(rng_key, inputs: EvaluationInputs):
    rng = foundry.random.PRNGSequence(rng_key)
    outputs = []
    for i in track(range(inputs.batches)):
        cond = foundry.random.uniform(next(rng), (inputs.batch_size, 2), minval=-2.5, maxval=2.5)
        rng_keys = foundry.random.split(next(rng), inputs.batch_size)
        output = F.vmap(evaluate_cond, in_axes=(None, 0, 0))(
            inputs, cond, rng_keys,
        )
        outputs.append(output)
    outputs, gen_data = tree.map(lambda *x: npx.concatenate(x, 0), *outputs)
    # train a network on the generated data to predict the alphas
    model = KeypointModel(
        tree.axis_size(inputs.keypoints, 0)
    )

    vars = model.init(next(rng), npx.zeros((2,)), npx.zeros((), dtype=npx.int32))
    iterations = 10_000
    optimizer = optax.adam(optax.cosine_decay_schedule(1e-3, iterations))
    opt_state = optimizer.init(vars["params"])

    @foundry.train.batch_loss
    def loss_fn(vars, rng_key, sample):
        alphas = model.apply(vars, sample.cond, sample.t)
        interpolated = alphas[:, None, None, None] * sample.out_keypoints
        interpolated = npx.sum(interpolated, axis=0)
        err = npx.mean(npx.square(interpolated - sample.out_model))
        return LossOutput(
            loss=err,
            metrics={"error": npx.sqrt(err)}
        )

    gen_data = tree.map(lambda x: npx.reshape(x, (-1,) + x.shape[2:]), gen_data)
    with foundry.train.loop(
        PyTreeData(
            gen_data
        ).stream().shuffle(next(rng)).batch(128),
        iterations=iterations,
        rng_key=next(rng)
    ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                opt_state, vars, metrics = foundry.train.step(
                    loss_fn, optimizer, opt_state=opt_state,
                    vars=vars, rng_key=step.rng_key,
                    batch=step.batch
                )
                if step.iteration % 1000 == 0:
                    foundry.train.console.log(
                        step.iteration, metrics
                    )
    def loss_fn(sample):
        alphas = model.apply(vars, sample.cond, sample.t)
        interpolated = alphas[:, None, None, None] * sample.out_keypoints
        interpolated = npx.sum(interpolated, axis=0)
        return npx.linalg.norm(interpolated - sample.out_model)

    lin_errors = jax.lax.map(loss_fn, gen_data, batch_size=64)
    lin_errors = npx.reshape(lin_errors, outputs.nw_error.shape)

    return EvaluationOutputs(
        outputs.cond,
        outputs.ts,
        outputs.nw_error,
        lin_errors,
        vars,
        inputs.keypoints,
    )

def run(config : Config):
    rng = foundry.random.PRNGSequence(config.seed)

    from .main import logger as main_logger
    main_logger.setLevel(logging.DEBUG)

    logger.setLevel(logging.DEBUG)
    logger.info(f"Evaluating {config}")

    wandb_run = wandb.init(
        project="image-diffusion-eval",
        config=tree.flatten_to_dict(config)[0]
    )
    api = wandb.Api()
    source_run = api.run(config.run)
    artifacts_list = list(source_run.logged_artifacts())
    artifacts_list = list(a for a in artifacts_list if a.type == "model")

    artifacts = {}
    for artifact in artifacts_list[:-1]:
        iterations = artifact.metadata["step"]
        if iterations % config.step_interval == 0:
            artifacts[iterations] = artifact
    artifacts[artifacts_list[-1].metadata["step"]] = artifacts_list[-1]
    final_step, final = list(artifacts.items())[-1]

    if config.only_final:
        artifacts = {final_step: final}

    path = Path(final.download()) / "checkpoint.zarr.zip"
    checkpoint = foundry.util.serialize.load_zarr(path)

    normalizer, train_data, _ = checkpoint.create_data()
    keypoints = jax.vmap(normalizer.normalize)(
        tree.map(lambda x: x[:16], train_data.as_pytree())
    ).cond

    model = checkpoint.config.create()
    schedule = checkpoint.schedule

    output_artifact = wandb.Artifact(name="evaluation", type="evaluation")
    for iterations, artifact in reversed(artifacts.items()):
        path = Path(artifact.download()) / "checkpoint.zarr.zip"
        checkpoint = foundry.util.serialize.load_zarr(path)
        logger.info(f"Evaluating iteration {iterations}")
        output = evaluate_checkpoint(next(rng), EvaluationInputs(
            vars=checkpoint.vars,
            model_apply=model.apply,
            schedule=schedule,
            keypoints=keypoints,

            sample_shape=train_data[0].data.shape,
            samples_per_cond=config.samples_per_cond,
            eval_per_cond=config.eval_per_cond,
            batch_size=config.batch_size,
            batches=config.batches,
            t_pct=config.t_pct
        ))
        output = replace(
            output,
            checkpoint_uri=artifact.get_entry("checkpoint.zarr.zip").ref
        )

        logger.info(f"NW percentiles: {npx.percentile(output.nw_error, npx.array([10, 20, 50, 70, 90]))}")
        logger.info(f"Linear percentiles: {npx.percentile(output.lin_error, npx.array([10, 20, 50, 70, 90]))}")
        logger.info(f"Linear error: {npx.mean(output.lin_error)}, NW error: {npx.mean(output.nw_error)}")
        result_url = f"{config.bucket_url}/{wandb_run.id}/{iterations:06}.zarr.zip"
        foundry.util.serialize.save(result_url, output)
        output_artifact.add_reference(result_url, f"eval_{iterations:06}.zarr.zip")
    wandb_run.log_artifact(output_artifact)
    output_artifact.wait()
    logger.info(f"Artifact: {output_artifact.source_qualified_name}")
    wandb_run.finish()

class KeypointModel(nn.Module):
    keypoints: int

    @nn.compact
    def __call__(self, cond, t):
        if t is not None: input = npx.concatenate([cond, t[None]])
        else: input = cond
        h = nn.Sequential([
            nn.Dense(64),
            nn.gelu,
            nn.Dense(64),
            nn.gelu
        ])(input)
        h = nn.Sequential([
            nn.Dense(64),
            nn.gelu,
            nn.Dense(64),
            nn.gelu,
            nn.Dense(self.keypoints),
        ])(npx.concatenate((h, t[None])))
        normalized = nn.softmax(h, axis=-1)
        return normalized