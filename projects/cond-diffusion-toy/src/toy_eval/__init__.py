from stanza.runtime import setup
setup()

from stanza.dataclasses import dataclass, replace
from stanza.runtime import ConfigProvider, command
from stanza.random import PRNGSequence
from stanza import canvas
from . import datasets
from .datasets import Sample

from functools import partial
from typing import Any

import stanza.policy
import stanza.util
import stanza.util
import stanza.train.reporting
import stanza.train.wandb
import jax
import jax.numpy as jnp
import functools
import wandb
import stanza
import plotly.express as px
import plotly.graph_objs as go
import logging
logger = logging.getLogger(__name__)

@dataclass
class Config:
    seed: int = 42
    dataset: str = "two_deltas"
    denoiser: str = "estimator"
    num_visualize_values: int = 32

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()

        from . import diffusion_estimator, diffusion_learned
        
        # Check for a default policy override
        denoiser = config.get("denoiser", str, default=None)
        if denoiser == "estimator":
            defaults = replace(defaults, denoiser=diffusion_estimator.DiffusionEstimatorConfig())
        elif denoiser == "learned":
            defaults = replace(defaults, denoiser=diffusion_learned.DiffusionLearnedConfig())
        else:
            defaults = replace(defaults, denoiser=diffusion_estimator.DiffusionEstimatorConfig())
        
        # Check for dataset override
        dataset = config.get("dataset", str, default=None)
        if dataset == "two_deltas":
            defaults = replace(defaults, dataset=datasets.TwoDeltasConfig())
        elif dataset == "two_delta_sequence":
            defaults = replace(defaults, dataset=datasets.TwoDeltaSequenceConfig())
        else:
            defaults = replace(defaults, dataset=datasets.TwoDeltasConfig())

        return config.get_dataclass(defaults)

def generate_plots(samples):
    fig = go.Figure()
    for i in range(samples.value.shape[-1]):
        fig.add_trace(go.Scatter(
            x=jnp.squeeze(samples.cond),
            y=samples.value[..., i], 
            mode='markers',
            name=f't = {i + 1}',
            marker=dict(
                color=f'rgba({i*256//samples.value.shape[-1]},0,255,255)',
                opacity=0.5
            )
        ))
    plots = {
        "samples": fig,
        #"transformed_samples": px.scatter(x=jnp.squeeze(samples.cond), y=jnp.squeeze(samples.value), opacity=0.5),
        #"norm": data_info.visualizer(samples, value_transform=lambda x: jnp.array([jnp.linalg.norm(x)])),
        #"score": plot_score(*denoised, value_transform=proj_transform),
        #"variances": plot_variance(samples)
        #"Wasserstein": plot_distance_ratio(samples, dist_name="OTdist"),
        #"TV": plot_distance_ratio(samples, dist_name="OTdist", cost_fn=TVCost(0.01), cost_transform=lambda x: x)
        #lambda samples: plot_distance_ratio(samples, dist_name="TVdist", intervals_per_window=16)
    }
    metrics = {
        #"avg_variance": jnp.mean(compute_variance(samples))
    }
    return metrics | plots 

def main(config : Config):
    logger.info(f"Running {config}")
    rng = PRNGSequence(jax.random.key(config.seed))

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = datasets.create(config.dataset, next(rng))
    train_data = dataset.splits["train"]
    test_data = dataset.splits["test"]
    train_sample = jax.tree_map(lambda x: x[0], train_data)

    wandb_run = wandb.init(
        project="cond_diffusion_toy",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    # denoiser: cond, rng_key -> Sample(cond, value)
    denoiser = config.denoiser.train_denoiser(
        wandb_run, train_data, next(rng)
    )

    logger.info(f"Performing final evaluation...")

    samples = jax.vmap(
        lambda cond, rng_key: jax.vmap(partial(denoiser, cond))(jax.random.split(rng_key, config.num_visualize_values))
    )(test_data.cond, jax.random.split(next(rng), test_data.cond.shape[0]))
    samples_cond = samples.cond.reshape(-1, train_sample.cond.shape[-1])
    samples_value = samples.value.reshape(samples_cond.shape[0], -1)
    samples = Sample(samples_cond, samples_value)

    if "visualize" in dataset.transforms:
        samples = replace(samples, value=jax.vmap(dataset.transforms["visualize"])(samples.value))
    output = generate_plots(samples)

    # get the metrics and final reportables
    # from the eval output
    # metrics, reportables = stanza.train.reporting.as_log_dict(output)
    # for k, v in metrics.items():
    #     logger.info(f"{k}: {v}")
    # wandb_run.summary.update(metrics)
    wandb_run.log(output)
    wandb_run.finish()

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    main(Config.parse(config))