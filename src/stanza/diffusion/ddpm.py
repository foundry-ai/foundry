from flax import struct
from functools import partial

import jax.numpy as jnp
import jax.flatten_util
import jax
import chex

@struct.dataclass
class DDPMSchedule:
    betas: jnp.array
    alphas: jnp.array = struct.field(default=None)
    alphas_cumprod: jnp.array = struct.field(default=None)
    variance_type: str = struct.field(default="fixed_small", pytree_node=False)
    # "epsilon", "sample", or "v_prediction"
    prediction_type: str = struct.field(default="epsilon", pytree_node=False)
    # If None, no clipping
    clip_sample_range: float = None

    def __post_init__(self):
        if self.alphas is None:
            object.__setattr__(self, 'alphas', 1 - self.betas)
        if self.alphas_cumprod is None:
            object.__setattr__(self, 'alphas_cumprod', jnp.cumprod(self.alphas))

    @staticmethod
    def make_linear(num_timesteps, beta_start=0.0001, beta_end=0.02,
                    **kwargs):
        return DDPMSchedule(
            betas=jnp.linspace(beta_start, beta_end, num_timesteps),
            **kwargs
        )
    
    @staticmethod
    def make_squaredcos_cap_v2(num_timesteps, max_beta=0.999, **kwargs):
        t1 = jnp.arange(num_timesteps).astype(float)/num_timesteps
        t2 = (jnp.arange(num_timesteps) + 1).astype(float)/num_timesteps
        def alpha_bar(t):
            return jnp.square(jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2))
        return DDPMSchedule(
            betas=jnp.minimum(1 - alpha_bar(t2) / alpha_bar(t1), max_beta), 
            **kwargs)

    @property
    def num_steps(self):
        return self.betas.shape[0]

    # This will do the noising
    # forward process
    # will return noisy_sample, noise_eps, model_output 
    @jax.jit
    def add_noise(self, rng_key, sample, timestep):
        sqrt_alphas_prod = jnp.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alphas_prod = jnp.sqrt(1 - self.alphas_cumprod[timestep])
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape)
        noisy_flat = sqrt_alphas_prod * sample_flat + \
            sqrt_one_minus_alphas_prod*noise_flat

        noisy = unflatten(noisy_flat)
        noise = unflatten(noise_flat)
        if self.prediction_type == "epsilon":
            return noisy, noise, noise
        elif self.prediction_type == "sample":
            return noisy, noise, sample

    @jax.jit
    def add_sub_noise(self, rng_key, sub_sample, sub_timestep, timestep):
        alphas_shifted = jnp.concatenate((jnp.ones((1,)), self.alphas_cumprod), axis=-1)
        alphas_prod = self.alphas_cumprod[timestep] / alphas_shifted[sub_timestep]

        sqrt_alphas_prod = jnp.sqrt(alphas_prod)
        sqrt_one_minus_alphas_prod = jnp.sqrt(1 - alphas_prod)
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sub_sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape)
        noisy_flat = sqrt_alphas_prod * sample_flat + \
            sqrt_one_minus_alphas_prod*noise_flat

        sqrt_one_minus_alphas_full_prod = jnp.sqrt(1- self.alphas_cumprod[timestep])
        scaling = sqrt_one_minus_alphas_prod / sqrt_one_minus_alphas_full_prod
        noisy = unflatten(noisy_flat)
        noise = unflatten(noise_flat)
        scaled_noise = unflatten(scaling*noise_flat)
        if self.prediction_type == "epsilon":
            return noisy, noise, scaled_noise
        elif self.prediction_type == "sample":
            return noisy, noise, sub_sample
    
    # This does a reverse process step
    @jax.jit
    def step(self, rng_key, sample, timestep, delta_steps, model_output):
        #chex.assert_trees_all_equal_shapes_and_dtypes(sample, model_output)
        t = timestep
        prev_t = t - delta_steps
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = jax.lax.cond(prev_t >= 0, lambda: self.alphas_cumprod[prev_t],
                                        lambda: jnp.ones(()))

        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_sample = (sample_flat - beta_prod_t ** (0.5) * model_output_flat) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_sample = model_output_flat
        elif self.prediction_type == "v_prediction":
            pred_sample = (alpha_prod_t**0.5) * sample_flat - (beta_prod_t**0.5) * model_output_flat

        if self.clip_sample_range is not None:
            pred_sample = jnp.clip(pred_sample, -self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = jnp.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
        current_sample_coeff = jnp.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_sample + current_sample_coeff * sample_flat

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = jnp.clip(variance, a_min=1e-20)
        sigma = jnp.sqrt(variance)
        noise = sigma*jax.random.normal(rng_key, pred_prev_sample.shape)
        return unflatten(pred_prev_sample + noise)

    def _sample_step(self, model, delta_steps, carry, timestep,
                     trajectory=False):
        rng_key, sample = carry
        rng_key, model_rng, step_rng = jax.random.split(rng_key, 3)
        with jax.named_scope("eval_model"):
            model_output = model(model_rng, sample, timestep)
        with jax.named_scope("step"):
            next_sample = self.step(step_rng, sample, timestep,
                                    delta_steps, model_output)
        out = next_sample if trajectory else None
        return (rng_key, next_sample), out

    # model is a map from rng_key, sample, timestep --> model_output
    def sample(self, rng_key, model, example_sample, *, num_steps=None,
                        final_step=None, static_loop=False,
                        trajectory=False):
        if final_step is None:
            final_step = self.num_steps
        if num_steps is None:
            num_steps = final_step
        step_ratio = final_step // num_steps
        step = partial(self._sample_step, model, step_ratio, trajectory=trajectory)
        # sample initial noise
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(example_sample)
        random_sample = unflatten(jax.random.normal(rng_key, sample_flat.shape))
        if trajectory:
            timesteps = (jnp.arange(0, num_steps) * step_ratio).round()[::-1] \
                    .copy().astype(jnp.int32)
            carry = (rng_key, random_sample)
            carry, out = jax.lax.scan(step, carry, timesteps)
            _, sample = carry
            out = jax.tree_map(lambda x, y: jnp.concatenate(
                                [jnp.expand_dims(x,axis=0), y], axis=0
                            ), random_sample, out)
            return sample, out
        else:
            # use a for loop
            def loop_step(i, carry):
                T = jnp.round((num_steps - 1 - i)*step_ratio)
                if static_loop:
                    carry, _ = jax.lax.cond(i < num_steps, 
                                step, lambda x,y: (x,None), carry, T)
                else:
                    carry, _ = step(carry, T)
                return carry
            carry = (rng_key, random_sample)
            carry = jax.lax.fori_loop(0, 
                self.num_steps if static_loop else num_steps, loop_step, carry)
            _, sample = carry
            return sample

    def loss(self, rng_key, model, 
             sample, t, *,
             model_has_state_updates=False):
        s_rng, m_rng = jax.random.split(rng_key)
        noised_sample, _, target = self.add_noise(s_rng, sample, t)
        pred = model(m_rng, noised_sample, t)
        if model_has_state_updates:
            pred, state = pred
        chex.assert_trees_all_equal_shapes_and_dtypes(pred, target)

        pred_flat = jax.flatten_util.ravel_pytree(pred)[0]
        target_flat = jax.flatten_util.ravel_pytree(target)[0]
        loss = jnp.mean((pred_flat - target_flat)**2)

        if model_has_state_updates:
            return loss, state
        else:
            return loss
