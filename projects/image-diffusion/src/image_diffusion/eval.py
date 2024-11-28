from foundry.core.dataclasses import dataclass
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.core import tree
from foundry.random import PRNGSequence

from typing import Any

import foundry.core as F

from pathlib import Path
from rich.progress import track

import foundry.util.serialize
import foundry.random
import jax
import foundry.numpy as npx

import wandb
import tempfile
import logging

logger = logging.getLogger("eval")

@dataclass
class Config:
    run : str = "dpfrommer-projects/image-diffusion/runs/1vpg8vbk"
    bucket_url: str = "s3://wandb-data"
    seed: int = 42

@dataclass
class EvaluationInputs:
    vars: Any
    model_apply: Any
    schedule: DDPMSchedule

@dataclass
class EvaluationOutputs:
    cond: jax.Array

    # These have an extra batch array of dim GENERATE_SAMPLES
    ts: jax.Array
    alphas: jax.Array
    nw_error: jax.Array
    lin_error: jax.Array

GENERATE_SAMPLES = 16
BATCH_SIZE = 16

@F.jit
def evaluate_cond(inputs: EvaluationInputs, cond : Any, rng_key : jax.Array):
    schedule = inputs.schedule

    # random keypoints to use...
    keypoints = foundry.random.normal(foundry.random.key(42), (16, 2))

    def sample(rng_key):
        r_rng, s_rng = foundry.random.split(rng_key)
        denoiser = lambda rng_key, x, t: inputs.model_apply(inputs.vars, x, t-1, cond=cond)
        sample, trajectory = schedule.sample(
            rng_key, denoiser, npx.zeros((28, 28, 1)), 
            trajectory=True
        )
        t = foundry.random.randint(s_rng, (), minval=0, maxval=trajectory.shape[0])
        return sample, trajectory[t], t

    rng_keys = foundry.random.split(rng_key, GENERATE_SAMPLES)
    samples, reverse_samples, ts = jax.lax.map(
        sample, rng_keys, 
        batch_size=8
    )

    def eval(eval_inputs):
        reverse_sample, t = eval_inputs
        model = lambda cond: inputs.model_apply(
            inputs.vars, reverse_sample, t-1, 
            cond=cond
        )
        keypoints_out = jax.lax.map(model, keypoints)
        model_out = model(cond)
        nw_out = schedule.output_from_denoised(
            reverse_sample, t,
            schedule.compute_denoised(
                reverse_sample, t, samples
            )
        )
        nw_err = npx.linalg.norm(nw_out - model_out)
        keypoints_out = keypoints_out.reshape(keypoints_out.shape[0], -1).T
        model_out = model_out.reshape(-1)
        alphas, lin_err, _, _ = npx.linalg.lstsq(keypoints_out, model_out)
        return alphas, nw_err, lin_err
    alphas, nw_errs, lin_errs = jax.lax.map(eval, (reverse_samples, ts))
    return EvaluationOutputs(
        cond, ts, nw_errs, lin_errs, alphas
    )


def evaluate_checkpoint(rng_key, inputs: EvaluationInputs):
    rng = foundry.random.PRNGSequence(rng_key)
    outputs = []
    for i in track(range(1)):
        cond = foundry.random.uniform(next(rng), (BATCH_SIZE, 2), minval=-2.5, maxval=2.5)
        rng_keys = foundry.random.split(next(rng), BATCH_SIZE)
        output = F.vmap(evaluate_cond, in_axes=(None, 0, 0))(
            inputs, cond, rng_keys
        )
        outputs.append(output)
    outputs = tree.map(lambda *x: npx.concatenate(x, 0), *outputs)
    return outputs

def run(config):
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
    artifacts = source_run.logged_artifacts()

    epochs_artifact = {}
    for artifact in artifacts:
        if artifact.type != "model": continue
        epoch = artifact.metadata["epochs"]
        epochs_artifact[epoch] = artifact

    path = Path(epochs_artifact[0].download()) / "checkpoint.zarr.zip"
    checkpoint = foundry.util.serialize.load_zarr(path)

    model = checkpoint.config.create()
    schedule = checkpoint.schedule

    denoiser = model.apply(
        checkpoint.vars, 
        npx.zeros((28,28,1)), 1, cond=npx.zeros((2,))
    )

    output_artifact = wandb.Artifact(name="evaluation", type="evaluation")
    with tempfile.TemporaryDirectory(prefix="eval") as directory:
        rng = foundry.random.PRNGSequence(config.seed)
        for epoch, artifact in epochs_artifact.items():
            path = Path(artifact.download()) / "checkpoint.zarr.zip"
            checkpoint = foundry.util.serialize.load_zarr(path)
            logger.info(f"Evaluating epoch {epoch}")
            output = evaluate_checkpoint(next(rng), EvaluationInputs(
                vars=checkpoint.vars,
                model_apply=model.apply,
                schedule=schedule
            ))
            logger.info(f"Linear error: {npx.mean(output.lin_error)}, NW error: {npx.mean(output.nw_error)}")
            result_url = f"{config.bucket_url}/{wandb_run.id}/{epoch:04}.zarr.zip"
            foundry.util.serialize.save(result_url, output)
            output_artifact.add_reference(result_url)
    wandb_run.log_artifact(output_artifact)
    wandb_run.finish()