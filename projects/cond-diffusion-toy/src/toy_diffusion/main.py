from foundry.core.dataclasses import dataclass
from foundry.util.registry import Registry
from foundry.datasets.core import Dataset
from foundry.diffusion.ddpm import DDPMSchedule
from foundry.data.core import PyTreeData

import foundry.random
import foundry.train
import foundry.numpy as npx
import foundry.core.tree as tree

from .data import Sample

import wandb
import math
import optax

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

    iterations: int = 20_000
    lr: float = 3e-4
    weight_decay: float = 1e-4

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

def train(config):
    logger.setLevel(logging.DEBUG)

    logger.info(f"Config: {config}")
    rng = foundry.random.PRNGSequence(config.seed)
    dataset = config.create_dataset()
    data = dataset.split("train")
    model_config = ModelConfig(model=config.model, sample_structure=data.structure)
    normalizer = dataset.normalizer(config.normalizer)
    model, vars = model_config.create_model(next(rng))

    wandb_run = wandb.init(
        project="toy-diffusion",
        config=tree.flatten_to_dict(config)[0]
    )

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps, prediction_type=config.prediction_type,
        clip_sample_range=2.
    )

    @foundry.train.batch_loss
    def loss_fn(vars, rng_key, sample):
        sample = normalizer.normalize(sample)
        diffuser = lambda rng_key, x, t: model.apply(vars, x, t, cond=sample.x)
        schedule.loss(rng_key, diffuser, sample.y)
    
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

    with foundry.train.loop(
        data.stream().shuffle(next(rng)).batch(128),
        iterations=config.iterations
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
                        step.iteration, metrics
                    )