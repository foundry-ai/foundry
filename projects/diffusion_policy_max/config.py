from diffusion_policy.diffusion_bc import Config as DiffusionConfig

def ant_diffusion():
    return DiffusionConfig(
        env="ant",
    )

def pendulum_diffusion():
    return DiffusionConfig(
        env="pendulum",
        net="mlp",
        features=[128, 128, 64, 64, 32],
        obs_horizon=1,
        action_horizon=1,
        action_padding=0,
        step_embed_dim=64,
    )

def quadrotor_diffusion():
    return DiffusionConfig(
        env="quadrotor",
        net="mlp",
        features=[128, 128, 64, 64, 32],
        obs_horizon=1,
        action_horizon=1,
        action_padding=0,
        step_embed_dim=64,
    )