
from stanza.data import Data
from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider

from stanza.dataclasses import dataclass
from stanza.diffusion import nonparametric
from .datasets import Sample

import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionEstimatorConfig:
    estimator: str = "nw"
    kernel_bandwidth: float = 0.01
    diffusion_steps: int = 50

    def parse(self, config: ConfigProvider) -> "DiffusionEstimatorConfig":
        return config.get_dataclass(self)

    def train_denoiser(self, wandb_run, train_data):
        return diffusion_estimator(self, wandb_run, train_data)

def diffusion_estimator(
            config: DiffusionEstimatorConfig,
            wandb_run,
            train_data : Data[Sample]
        ):
    data_sample = jax.tree_map(lambda x: x[0], train_data)
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )
    def denoiser(cond, rng_key) -> Sample:
        data = train_data.cond, train_data.value
        if config.estimator == "nw":
            kernel = nonparametric.log_gaussian_kernel
            estimator = lambda cond: nonparametric.nw_cond_diffuser(
                cond, data, schedule, kernel, config.kernel_bandwidth
            )
        diffuser = estimator(cond)
        value = schedule.sample(rng_key, diffuser, data_sample.value)
        return Sample(cond, value)
    return denoiser