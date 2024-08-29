from foundry.core.dataclasses import dataclass

from .methods.behavior_cloning import BCConfig
from .methods.diffusion_estimator import EstimatorConfig
from .methods.diffusion_policy import DPConfig
from .methods.nearest_neighbor import NearestConfig

from .common import MethodConfig

from omegaconf import MISSING

@dataclass
class Config:
    seed: int = 42
    # these get mixed into the "master" seed
    train_seed: int = 42
    eval_seed: int = 42

    method : str = "diffusion_policy"
    dataset : str = "robomimic/pickplace/can/ph"

    # total trajectories to load
    train_trajectories : int | None = None

    eval_trajectories : int = 4
    render_trajectories : int = 4

    obs_length: int = 1
    action_length: int = 16

    timesteps: int = 200

    render_width = 128
    render_height = 128

    bc : BCConfig = BCConfig()
    estimator : EstimatorConfig = EstimatorConfig()
    dp: DPConfig = DPConfig()
    nearest: NearestConfig = NearestConfig()

    @property
    def method_config() -> MethodConfig:
        match self.method:
            case "bc": return self.bc
            case "estimator": return self.estimator
            case "diffusion_policy": return self.dp
            case "nearest": return self.nearest
            case _: raise ValueError(f"Unknown method: {self.method}")