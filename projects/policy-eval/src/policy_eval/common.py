import foundry.core as F
import foundry.numpy as jnp

from foundry.random import PRNGSequence

from foundry.core.dataclasses import dataclass
from foundry.core.typing import Array
from foundry.data import Data
from foundry.env import Environment
from foundry.policy import Policy
from foundry.train.reporting import Video

from wandb.sdk.wandb_run import Run # type: ignore

from typing import Any, Callable

@dataclass
class Sample:
    state: Any
    observations: Array
    actions: Array

@dataclass
class Inputs:
    wandb_run: Run
    timesteps: int
    rng: PRNGSequence
    env: Environment

    # Will rollout specifically on the validation dataset
    # returns normalized rewards, raw rewards, 
    # as well as a grid Video
    # (useful for logging i.e. during training)
    validate : Callable[[Array, Policy], Array]
    validate_render: Callable[[Array, Policy], tuple[Array, Video]]

    train_data : Data[Sample]
    test_data : Data[Sample]

# Should be a PyTree result that can be saved.
class Result:
    def create_policy(self) -> Policy:
        pass

class MethodConfig:
    def run(self, inputs: Inputs) -> Result:
        raise NotImplementedError()