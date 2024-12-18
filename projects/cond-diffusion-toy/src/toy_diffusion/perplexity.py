from foundry.core.dataclasses import dataclass
from foundry.util.registry import Registry
from foundry.datasets.core import Dataset
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.data.core import PyTreeData
from foundry.train import LossOutput

import foundry.core as F
import foundry.random
import foundry.train
import foundry.train.console
import foundry.train.wandb
import foundry.numpy as npx
import foundry.core.tree as tree

from .data import Sample

import wandb
import math
import optax
import jax
import plotly.graph_objects as go
import flax.linen as nn

import logging
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model: str = "diffusion/mlp/small"
    seed: int = 42

    dim: int = 16

    base_deltas: int = 1
    perplexity_diff: int = 8

    timesteps: int = 32
    prediction_type: str = "epsilon"

    iterations: int = 10_000
    lr: float = 3e-4
    weight_decay: float = 1e-5

    def create_dataset(self):
        from .data import register_all
        registry = Registry[Dataset]()
        register_all(registry)
        return registry.create(self.dataset)

@dataclass
class ModelConfig:
    model: str
    sample_structure: Sample

    def create_model(self, rng_key=None):
        from foundry.models import register_all
        registry = Registry()
        register_all(registry)
        model = registry.create(self.model)
        if rng_key is None:
            return model
        vars = model.init(rng_key, tree.map(
            lambda x: npx.zeros_like(x), self.sample_structure.y
        ), t=npx.zeros((), dtype=npx.int32), cond=tree.map(
            lambda x: npx.zeros_like(x), self.sample_structure.x
        ))
        return model, vars

@dataclass
class Checkpoint:
    config: Config
    model_config: ModelConfig
    schedule: DDPMSchedule
    step: int
    vars: dict

def create_data(rng_key, dim, left_deltas, right_deltas):
    l_rng, r_rng, ln_rng, rn_rng = foundry.random.split(rng_key, 4)

    left_locs = foundry.random.normal(l_rng, (left_deltas, dim)).clip(-0.5, 0.5)
    right_locs = foundry.random.normal(r_rng, (right_deltas, dim)).clip(-0.5, 0.5)

    left_ys = left_locs + 0.05*foundry.random.normal(ln_rng, (128, left_deltas, dim))
    left_ys = left_ys.reshape((-1, dim))
    right_ys = right_locs + 0.05*foundry.random.normal(rn_rng, (128, right_deltas, dim))
    right_ys = right_ys.reshape((-1, dim))

    left_xs = npx.zeros((left_ys.shape[0],))
    right_xs = npx.ones((right_ys.shape[0],))
    return PyTreeData(
        Sample(
            x=npx.concatenate((left_xs, right_xs), 0),
            y=npx.concatenate((left_ys, right_ys), 0)
        )
    )

def train(config):
    logger.setLevel(logging.DEBUG)

    logger.info(f"Config: {config}")
    rng = foundry.random.PRNGSequence(config.seed)

    data = create_data(next(rng), config.dim, 
        config.base_deltas, config.base_deltas + config.perplexity_diff
    )

    model_config = ModelConfig(model=config.model, sample_structure=data.structure)
    model, vars = model_config.create_model(next(rng))

    keypoints = npx.array([0., 1.])

    wandb_run = wandb.init(
        project="toy-diffusion",
        config=tree.flatten_to_dict(config)[0]
    )
    logger.info(f"Run: {wandb_run.url}")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps, prediction_type=config.prediction_type,
        clip_sample_range=2.
    )

    def create_nw_denoiser(vars, cond):
        def diffuser(rng_key, x, t):
            return model.apply(vars, x, t, cond=cond)
        sample = lambda r: schedule.sample(r, diffuser, data.structure.y)
        samples = jax.vmap(sample)(foundry.random.split(next(rng), 128))
        def denoiser(rng_key, x_noised, t):
            return schedule.output_from_denoised(
                x_noised, t, 
                schedule.compute_denoised(x_noised, t, samples)
            )
        return denoiser

    @foundry.train.batch_loss
    def loss_fn(vars, rng_key, sample):
        n_rng, t_rng = foundry.random.split(rng_key)
        t = jax.random.randint(t_rng, (), 0, schedule.num_steps) + 1
        noised_sample_y, _, target = schedule.add_noise(n_rng, sample.y, t)
        pred = model.apply(vars, noised_sample_y, t, cond=sample.x)
        loss = npx.mean((pred - target)**2)

        return LossOutput(
            loss=loss, metrics={"loss": loss}
        )
    
    data = data.as_pytree()
    # amplify the data so there are at least 1024 samples
    amp = math.ceil(1024 / tree.axis_size(data, 0))
    data = tree.map(lambda x: npx.repeat(x, amp, axis=0), data)
    data = PyTreeData(data)

    iterations = config.iterations
    lr_schedule = optax.cosine_onecycle_schedule(
        iterations, config.lr, pct_start=0.02
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    opt_state = optimizer.init(vars["params"])

    @jax.jit
    def evaluate(rng_key, vars, x):
        t = schedule.num_steps // 2
        # samples
        s_rng, e_rng = foundry.random.split(rng_key)

        # sample from the NN
        denoiser = lambda rng_key, y, t: model.apply(
            vars, y, t, cond=x
        )
        nw_samples = jax.vmap(
            lambda r: schedule.sample(r, denoiser, npx.zeros((config.dim,)))
        )(
            foundry.random.split(s_rng, 16*1024)
        )
        def eval(y):
            nw_output = schedule.output_from_denoised(
                y, t,
                schedule.compute_denoised(y, t, nw_samples)
            )
            nn_output = denoiser(None, y, t)
            return npx.linalg.norm(nw_output - nn_output)
        y = foundry.random.uniform(rng_key, (1024, config.dim), minval=-2, maxval=2)
        return npx.mean(jax.vmap(eval)(y))

    checkpoints = []
    with foundry.train.loop(
        data.stream().shuffle(next(rng)).batch(256),
        rng_key=next(rng),
        iterations=config.iterations, show_epochs=False
    ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                opt_state, vars, grad_norm, metrics = foundry.train.step(
                    loss_fn, optimizer, opt_state=opt_state,
                    vars=vars, rng_key=step.rng_key,
                    batch=step.batch,
                    return_grad_norm=True
                )
                foundry.train.wandb.log(
                    step.iteration, metrics, {"lr": lr_schedule(step.iteration), "grad_norm": grad_norm},
                    run=wandb_run, prefix="train/"
                )
                if step.iteration % 1000 == 0:
                    average_nw_error = evaluate(next(rng), vars, 0.5)
                    nw_error_left = evaluate(next(rng), vars, 0.)
                    nw_error_right = evaluate(next(rng), vars, 1.)
                    eval_metrics = {
                        "nw_error": average_nw_error,
                        "nw_error_left": nw_error_left,
                        "nw_error_right": nw_error_right
                    }
                    foundry.train.console.log(
                        step.iteration, metrics, eval_metrics,
                    )
                    foundry.train.wandb.log(
                        step.iteration, eval_metrics,
                        run=wandb_run, prefix="test/"
                    )
                if step.iteration % 1000 == 0:
                    checkpoint = Checkpoint(
                        step=step.iteration,
                        config=config,
                        model_config=model_config,
                        schedule=schedule,
                        vars=tree.map(lambda x: npx.copy(x), vars)
                    )
                    checkpoints.append(checkpoint)
    # create final samples
    denoiser = lambda rng_key, y, t: model.apply(vars, y, t, cond=0.5)
    nw_samples = jax.vmap(
        lambda r: schedule.sample(r, denoiser, npx.zeros((config.dim,)))
    )(foundry.random.split(next(rng), 16*1024))

    # compute the "retrospective" validation loss
    def compute_val_loss(vars, sample_y):
        schedule.loss()

    wandb_run.summary["nw_error"] = average_nw_error