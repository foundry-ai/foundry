from .models.conditional_unet1d_jax import CondUNet1d
import dill
import torch
from foundry.datasets.util import cache_path

from ..common import Result, Inputs, DataConfig
from typing import Callable
import foundry.util
from foundry.data import Data
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env.core import Environment
from foundry.core import tree

from foundry.core.dataclasses import dataclass
from foundry.diffusion import DDPMSchedule
from foundry.data.normalizer import Normalizer, LinearNormalizer, StdNormalizer
from foundry.train import Vars

import jax
import foundry.numpy as jnp
import logging
logger = logging.getLogger(__name__)


@dataclass
class Pretrained(Result):
    data: DataConfig # dataset this model was trained on
    observations_structure: tuple[int]
    actions_structure: tuple[int]
    action_horizon: int

    schedule: DDPMSchedule

    obs_normalizer: Normalizer
    action_normalizer: Normalizer

    model: Callable
    vars: Vars

    def create_denoiser(self):
        return lambda obs, rng_key, noised_actions, t: self.model(
            self.vars, rng_key, obs, noised_actions, t - 1
        )

    def create_policy(self) -> Policy:
        # TODO: assert that the vars are the same type/shape
        def chunk_policy(input: PolicyInput) -> PolicyOutput:
            obs = input.observation
            obs = self.obs_normalizer.normalize(obs)
            model_fn = lambda rng_key, noised_actions, t: self.model(
                self.vars, noised_actions, t - 1, global_cond=obs
            )
            action = self.schedule.sample(input.rng_key, model_fn, self.actions_structure) 
            action = self.action_normalizer.unnormalize(action)
            return PolicyOutput(action=action[:self.action_horizon], info=action)
        obs_horizon = tree.axis_size(self.observations_structure, 0)
        return ChunkingTransform(
            obs_horizon, self.action_horizon
        ).apply(chunk_policy)

@dataclass
class PretrainedConfig:
    model_name: str = "low_dim/can_ph/diffusion_policy_cnn/train_0"

    diffusion_steps: int = 32
    action_horizon: int = 16

    def run(self, inputs: Inputs):
        _, data = inputs.data.load({"train", "test"})
        logger.info("Materializing all data...")
        train_data = data["train"].cache()
        test_data = data["test"].cache()

        schedule = DDPMSchedule.make_squaredcos_cap_v2(
            self.diffusion_steps,
            prediction_type="epsilon",
            clip_sample_range=1.0
        )
        train_sample = train_data[0]
        observations_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.observations)
        actions_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.actions)
        logger.info(f"Observation: {observations_structure}")
        logger.info(f"Action: {actions_structure}")

        #def get_jax_model(name):
            # path = cache_path(name, "latest.ckpt")
            # pt_model = ConditionalUnet1D(
            #     input_dim=10,
            #     local_cond_dim=None,
            #     global_cond_dim=46,
            #     diffusion_step_embed_dim=256,
            #     down_dims=[256,512,1024],
            #     kernel_size=5,
            #     n_groups=8,
            #     cond_predict_scale=True
            # )
            # pt_checkpoint = torch.load(open(path, 'rb'), pickle_module=dill)
            # #print(pt_checkpoint.keys())
            # state_dict = {}
            # for (k, v) in pt_checkpoint['state_dicts']['ema_model'].items():
            #     if k.startswith('model.'):
            #         k = k[len('model.'):]
            #         state_dict[k] = v
            # pt_model.load_state_dict(state_dict)
            # jax_model = t2j(pt_model)
            # params = {k: t2j(v) for k,v in pt_model.named_parameters()}
            # return jax_model, params
        
        model = CondUNet1d(
            input_dim=10,
            local_cond_dim=None,
            global_cond_dim=46,
            diffusion_step_embed_dim=256,
            down_dims=[256,512,1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

        params_path = cache_path(self.model_name, "params.npy")
        params = jax.jit(model.init)(next(inputs.rng), train_sample.actions, 0, global_cond=train_sample.observations)
        pretrained_params = jnp.load(params_path, allow_pickle=True).item()
        for k in params.keys():
            params[k] = pretrained_params[k]

        #total_params = sum(v.size for v in tree.leaves(vars))

        #logger.info(f"Total parameters: {total_params}")

        #TODO: replace with stats from ckpt
        normalizer = StdNormalizer.from_data(train_data)

        return Pretrained(
            data=inputs.data,
            observations_structure=observations_structure,
            actions_structure=actions_structure,
            action_horizon=self.action_horizon,
            schedule=schedule,
            obs_normalizer=normalizer.map(lambda x: x.observations),
            action_normalizer=normalizer.map(lambda x: x.actions),
            model=model,
            vars=params
        )