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
    dataset: str = "three_deltas"
    normalizer: str = "identity"
    model: str = "diffusion/mlp/small"

    seed: int = 42

    timesteps: int = 32
    prediction_type: str = "epsilon"

    iterations: int = 4_000
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
            lambda x: npx.zeros_like(x), self.sample_structure.y
        ))
        return model, vars

@dataclass
class Checkpoint:
    config: Config
    model_config: ModelConfig
    schedule: DDPMSchedule
    step: int
    vars: dict


class AlphaModel(nn.Module):
    outputs: int = 2
    @nn.compact
    def __call__(self, x, t=None):
        if t is not None:
            input = npx.concatenate([x[None], npx.expand_dims(t, -1)], axis=-1)
        else:
            input = x[None]
        out = nn.Sequential([
            nn.Dense(32),
            nn.relu,
            nn.Dense(32),
            nn.relu,
            nn.Dense(self.outputs),
        ])(input)
        return jax.nn.softmax(out)

def train(config):
    logger.setLevel(logging.DEBUG)

    logger.info(f"Config: {config}")
    rng = foundry.random.PRNGSequence(config.seed)
    dataset = config.create_dataset()
    data = dataset.split("train")
    model_config = ModelConfig(model=config.model, sample_structure=data.structure)
    normalizer = dataset.normalizer(config.normalizer)
    model, vars = model_config.create_model(next(rng))

    keypoints = dataset.keypoints
    alphas_model = AlphaModel(len(keypoints))

    wandb_run = wandb.init(
        project="toy-diffusion",
        config=tree.flatten_to_dict(config)[0]
    )
    logger.info(f"Run: {wandb_run.url}")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps, prediction_type=config.prediction_type,
        clip_sample_range=2.
    )

    @F.jit
    def generate_samples(vars, rng_key):
        def generate_sample(vars, cond, rng_key):
            diffuser = lambda rng_key, x, t: model.apply(vars, x, t, cond=cond)
            sample = schedule.sample(rng_key, diffuser, data.structure.y)
            sample = Sample(x=cond, y=sample)
            return normalizer.unnormalize(sample)
        cond = foundry.random.uniform(rng_key, (128*128*4,))
        samples = jax.vmap(generate_sample, in_axes=(None, 0, 0))(
            vars, cond, foundry.random.split(rng_key, cond.shape[0])
        )
        return samples

    @F.jit
    def generate_linear_samples(vars, rng_key):
        def generate_sample(cond, rng_key):
            def diffuser(rng_key, x, t):
                keypoints_out = jax.lax.map(
                    lambda k: model.apply(vars, x, t, cond=k),
                    keypoints
                )
                dist = npx.abs(cond - keypoints)
                args = npx.argsort(dist)
                i, j = args[0], args[1]
                a, b = dist[i], dist[j]
                alpha = b / (a + b)
                return alpha*keypoints_out[i] + (1 - alpha)*keypoints_out[j]
            sample = schedule.sample(rng_key, diffuser, data.structure.y)
            sample = Sample(x=cond, y=sample)
            return normalizer.unnormalize(sample)
        cond = foundry.random.uniform(rng_key, (128*128*4,))
        samples = jax.vmap(generate_sample)(
            cond, foundry.random.split(rng_key, cond.shape[0])
        )
        return samples

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
        sample = normalizer.normalize(sample)

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
                    foundry.train.console.log(
                        step.iteration, metrics,
                    )
                    samples = generate_samples(vars, next(rng))
                    samples_fig = go.Figure([
                        go.Histogram2d(x=samples.x, y=samples.y, nbinsx=128, nbinsy=128,
                                       histnorm="probability", colorscale=["white", "blue"])
                    ], layout=dict(scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y"))))
                    linear_samples = generate_linear_samples(vars, next(rng))
                    linear_samples_fig = go.Figure([
                        go.Histogram2d(x=linear_samples.x, y=linear_samples.y, nbinsx=128, nbinsy=128,
                                       histnorm="probability", colorscale=["white", "blue"])
                    ], layout=dict(scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y"))))
                    wandb_run.log(dict(
                        samples=samples_fig, linear_samples=linear_samples_fig,
                    ), step=step.iteration)

                    nw_errs = {}
                    def visualize_timestep(t):
                        xs = npx.linspace(0., 1., 128)
                        ys = npx.linspace(-1.25, 1.25, 128)

                        def evaluate(x, ys):
                            denoiser = create_nw_denoiser(vars, x)
                            return jax.vmap(denoiser, in_axes=(None, 0, None))(None, ys, t)
                        nw_z = jax.vmap(evaluate, in_axes=(0, None))(xs, ys).T

                        xs, ys = npx.meshgrid(xs, ys)
                        nn_z = jax.vmap(jax.vmap(
                            lambda x, y: model.apply(vars, y, t, cond=x)
                        ))(xs, ys)

                        def linear_evaluate(cond, y_noised):
                            keypoints_out = jax.lax.map(
                                lambda k: model.apply(vars, y_noised, t, cond=k),
                                keypoints
                            )
                            dist = npx.abs(cond - keypoints)
                            args = npx.argsort(dist)
                            i, j = args[0], args[1]
                            a, b = dist[i], dist[j]
                            alpha = b / (a + b)
                            return alpha*keypoints_out[i] + (1 - alpha)*keypoints_out[j]
                        lin_z = jax.vmap(jax.vmap(linear_evaluate))(xs, ys)

                        # nn_zs = 
                        nn_trace = go.Surface(x=xs, y=ys, z=nn_z, colorscale="Viridis", name="nn") 
                        nw_trace = go.Surface(x=xs, y=ys, z=nw_z, colorscale="Plasma", name="nw") 
                        lin_trace = go.Surface(x=xs, y=ys, z=lin_z, colorscale="Blues", name="lin") 
                        nw_nn_diff = go.Surface(x=xs, y=ys, z=nn_z - nw_z)
                        nw_lin_diff = go.Surface(x=xs, y=ys, z=nn_z - lin_z)

                        return {
                            "denoiser": go.Figure([nn_trace, nw_trace, lin_trace], layout=dict(
                                scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y_noised"), zaxis=dict(title="epsilon"))
                            )),
                            "nw_nn_diff": go.Figure(
                                [nw_nn_diff], layout=dict(
                                    scene=dict(xaxis_title="x", yaxis_title="y_noised", zaxis_title="epsilon_diff")
                            )),
                            "nw_lin_diff": go.Figure(
                                [nw_lin_diff], layout=dict(
                                    scene=dict(xaxis_title="x", yaxis_title="y_noised", zaxis_title="epsilon_diff")
                            )),
                            "abs_diff": go.Figure(
                                [go.Surface(x=xs, y=ys, z=npx.abs(nn_z - nw_z), colorscale="Viridis", name="nn_nw_diff")], layout=dict(
                                    scene=dict(xaxis_title="x", yaxis_title="y_noised", zaxis_title="epsilon_diff")
                            )),
                        }

                    wandb_run.log({
                        "denoiser_t.5/graphs": visualize_timestep(5),
                        "denoiser_t.10/graphs": visualize_timestep(10),
                        "denoiser_t.20/graphs": visualize_timestep(20),
                        "nw_errors_t": nw_errs
                    })