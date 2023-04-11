from stanza.util.dataclasses import dataclass, field

import jax.numpy as jnp

@dataclass(jax=True)
class DDPMSchedule:
    betas: jnp.array
    alphas: jnp.array = field(init=False)

    def __postinit__(self):
        self.alphas = 1 - self.betas

    @staticmethod
    def make_linear(num_timesteps, beta_start=0.0001, beta_end=0.02):
        return DDPMSchedule(
            betas=jnp.linspace(beta_start, beta_end, num_timesteps)
        )