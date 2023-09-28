from diffusion_policy import Config

def ant():
    return Config(
        data="ant",
    )

def pendulum():
    return Config(
        data="pendulum",
        net="mlp",
        features=[128, 128, 64, 64, 32],
        obs_horizon=1,
        action_horizon=1,
        diffusion_horizon=1,
        step_embed_dim=64,
    )

def quadrotor():
    return Config(
        data="quadrotor",
        net="mlp",
        features=[128, 128, 64, 64, 32],
        obs_horizon=1,
        action_horizon=1,
        diffusion_horizon=1,
        step_embed_dim=64,
    )