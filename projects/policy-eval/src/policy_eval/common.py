import boto3.s3
import foundry.core as F
import foundry.numpy as jnp
import foundry.datasets.env
import foundry.util.serialize
import tempfile
import urllib.parse

from foundry.random import PRNGSequence

from foundry.core import tree
from foundry.core.dataclasses import dataclass
from foundry.core.typing import Array
from foundry.datasets.env import EnvDataset
from foundry.datasets.core import DatasetRegistry
from foundry.data import Data
from foundry.env.core import Environment, ObserveConfig
from foundry.policy import Policy
from foundry.train.reporting import Video

from typing import Any, Callable
from pathlib import Path
from wandb.sdk.wandb_run import Run # type: ignore

from .lower_bounds import stable

import boto3
import logging

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    state: Any
    observations: Array
    actions: Array

@dataclass
class DataConfig:
    dataset: str
    env_type: str # usually "positional", "keypoint", or "rel_keypoint"
    train_trajectories: int | None
    test_trajectories: int | None
    validation_trajectories: int | None

    action_length: int
    obs_length: int

    def action_observation(self) -> ObserveConfig:
        if self.dataset.startswith("robomimic"):
            from foundry.env.mujoco.robosuite import EEfPose
            return EEfPose()
        elif self.dataset.startswith("pusht"):
            from foundry.env.mujoco.pusht import PushTAgentPos
            return PushTAgentPos()

    def _process_data(self, env : Environment, data):
        action_obs = self.action_observation()
        def process_element(element):
            if element.state is None: return env.full_state(element.reduced_state)
            else: return element.state
        data = data.map_elements(process_element).cache()
        data = data.chunk(
            self.action_length + self.obs_length
        )
        def process_chunk(chunk):
            states = chunk.elements
            actions = F.vmap(lambda s: env.observe(s, action_obs))(states)
            actions = tree.map(lambda x: x[-self.action_length:], actions)
            obs_states = tree.map(lambda x: x[:self.obs_length], states)
            curr_state = tree.map(lambda x: x[-1], obs_states)
            obs = F.vmap(env.observe)(obs_states)
            return Sample(
                curr_state, obs, actions
            )
        data = data.map(process_chunk)
        return data
    
    def load(self, splits=set()) -> tuple[Environment, dict[str, Data[Sample]]]:
        datasets = DatasetRegistry[EnvDataset]()

        foundry.datasets.env.register_all(datasets)
        stable.register_datasets(datasets)

        dataset = datasets.create(self.dataset)
        env = dataset.create_env(type=self.env_type)
        loaded_splits = {}
        if "train" in splits:
            logger.info(f"Loading training data from [blue]{self.dataset}[/blue]")
            train_data = dataset.split("train")
            if self.train_trajectories is not None:
                train_data = train_data.slice(0, self.train_trajectories)
            train_data = self._process_data(env, train_data)
            loaded_splits["train"] = train_data
        if "test" in splits:
            logger.info(f"Loading test data from [blue]{self.dataset}[/blue]")
            test_data = dataset.split("test")
            if self.test_trajectories is not None:
                test_data = test_data.slice(0, self.test_trajectories)
            test_data = self._process_data(env, test_data)
            loaded_splits["test"] = test_data
        if "validation" in splits:
            logger.info(f"Loading validation data from [blue]{self.dataset}[/blue]")
            validation_data = dataset.split("validation")
            if self.validation_trajectories is not None:
                validation_data = validation_data.slice(0, self.validation_trajectories)
            # get the first state of the trajectory
            validation_data = validation_data.truncate(1).map(
                lambda x: env.full_state(tree.map(lambda y: y[0], x.reduced_state))
            )
            loaded_splits["validation"] = validation_data
        return env, loaded_splits

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

    bucket_url : str | None
    data : DataConfig

# Should be a PyTree result that can be saved.
class Result:
    def create_policy(self) -> Policy:
        pass

    def save(self, path: Path | str):
        path = Path(path)
        foundry.util.serialize.save_zarr(path, self, None)
    
    def save_s3(self, s3_url : str):
        s3_client = boto3.client("s3")
        parsed = urllib.parse.urlparse(s3_url)
        assert parsed.scheme == "s3"
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / "result.zarr.zip"
            self.save(path)
            s3_client.upload_file(path, bucket, key)

    @staticmethod
    def load(path: Path | str) -> 'Result':
        path = Path(path)
        result, _ = foundry.util.serialize.load_zarr(path)
        return result
    
    @staticmethod
    def load_s3(s3_url : str):
        s3_client = boto3.client("s3")
        parsed = urllib.parse.urlparse(s3_url)
        assert parsed.scheme == "s3"
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / "result.zarr.zip"
            s3_client.download_file(bucket, key, path)
            return Result.load(path)

class MethodConfig:
    def run(self, inputs: Inputs) -> Result:
        raise NotImplementedError()