from foundry.core.dataclasses import dataclass, field

import foundry.core as F
import foundry.core.tree as tree

import foundry.util
import foundry.numpy as jnp
import jax.flatten_util
import jax
import chex

from functools import partial
from typing import Optional, TypeVar, Callable

Sample = TypeVar("Sample")

@dataclass
class DDPMSchedule:
    """A schedule for a DDPM model. Implements https://arxiv.org/abs/2006.11239. """
    betas: jax.Array
    """ The betas for the DDPM. This corresponds to the forward process:

            q(x_t | x_{t-1}) = N(x_t | sqrt(1 - beta_t)x_{t-1}, beta_t I)

       Note that betas[1] corresponds to beta_1 and betas[T] corresponds to beta_T.
       betas[0] should always be 0.
    """
    alphas: jax.Array
    """ 1 - betas """
    alphas_cumprod: jax.Array
    """ The alphabar_t for the DDPM. alphabar_t = prod_(i=1)^t (1 - beta_i)
    Note that:

        alphas_cumprod[0] = alphabar_0 = 1

        alphas_cumprod[1] = alphabar_1 = alpha_1 = (1 - beta_1)

    """
    prediction_type: str = "epsilon"
    """ The type of prediction to make. If "epsilon", the model will predict the noise.
    If "sample", the model will predict the sample.
    """
    clip_sample_range: Optional[float] = None
    """ Whether to clip the predicted denoised sample in the reverse process to the range [-clip_sample_range, clip_sample_range].
       If None, no clipping is done.
    """

    @staticmethod
    def make_from_betas(betas: jax.Array, **kwargs) -> "DDPMSchedule":
        alphas = 1 - betas
        alphas_cumprod = jnp.cumprod(alphas)
        return DDPMSchedule(
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            **kwargs
        )

    @staticmethod
    def make_from_alpha_bars(alphas_cumprod : jax.Array, max_beta : float = 1., **kwargs) -> "DDPMSchedule":
        """ Makes a DDPM schedule from the alphas_cumprod. """
        t1 = jnp.roll(alphas_cumprod, 1, 0)
        t1 = t1.at[0].set(1)
        t2 = alphas_cumprod
        betas = 1 - t2/t1
        betas = jnp.clip(betas, 0, max_beta)
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs)

    @staticmethod
    def make_linear(num_timesteps : int, beta_start : float = 0.0001, beta_end : float = 0.1,
                    **kwargs) -> "DDPMSchedule":
        beta_end = jnp.clip(beta_end, 0, 1)
        beta_start = jnp.clip(beta_start, 0, 1)
        betas = jnp.linspace(beta_start, beta_end, num_timesteps, dtype=jnp.float32)
        betas = jnp.concatenate((jnp.zeros((1,), jnp.float32), betas))
        """ Makes a linear schedule for the DDPM. """
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs
        )
    
    @staticmethod
    def make_rescaled(num_timesteps, schedule, **kwargs):
        """ Rescales a schedule to have a different number of timesteps. """
        xs = jnp.linspace(0, 1, num_timesteps + 1, dtype=jnp.float32)
        old_xs = jnp.linspace(0, 1, schedule.num_steps + 1, dtype=jnp.float32)
        new_alphas_cumprod = jnp.interp(xs, old_xs, schedule.alphas_cumprod)
        return DDPMSchedule.make_from_alpha_bars(new_alphas_cumprod, **kwargs)
    
    @staticmethod
    def make_squaredcos_cap_v2(num_timesteps : int, order: float = 2, offset : float | None = None, max_beta : float = 0.999, **kwargs) -> "DDPMSchedule":
        """ Makes a squared cosine schedule for the DDPM.
            Uses alpha_bar(t) = cos^2((t + 0.008) / 1.008 * pi / 2)
            i.e. the alpha_bar is a squared cosine function.

            This means a large amount of noise is added at the start, with
            decreasing noise added as time goes on.
        """
        t = jnp.arange(num_timesteps, dtype=jnp.float32)/num_timesteps
        offset = offset if offset is not None else 0.008
        def alpha_bar(t):
            t = (t + offset) / (1 + offset)
            return jnp.pow(jnp.cos(t * jnp.pi / 2), order)
        # make the first timestep start at index 1
        alpha_bars = jnp.concatenate(
            (jnp.ones((1,), dtype=t.dtype), jax.vmap(alpha_bar)(t)),
        axis=0)
        # alpha_bars = alpha_bars.at[-1].set(0)
        return DDPMSchedule.make_from_alpha_bars(alpha_bars, max_beta=max_beta, **kwargs)
    
    @staticmethod
    def make_scaled_linear_schedule(num_timesteps : int,
                beta_start : float = 0.0001, beta_end : float = 0.02, **kwargs):
        """ Makes a scaled linear (i.e quadratic) schedule for the DDPM. """
        betas = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float32),
            jnp.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=jnp.float32)**2),
        axis=-1)
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs
        )

    @property
    def reverse_variance(self):
        alpha_bars = self.alphas_cumprod[1:]
        alpha_bars_prev = self.alphas_cumprod[:-1]
        betas = self.betas[1:]
        variance = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas
        variance = jnp.concatenate((jnp.zeros((1,)), variance), axis=0)
        return variance

    @property
    def num_steps(self) -> int:
        """ The number of steps T in the schedule. Note that betas has T+1 elements since beta_0 = 0."""
        return self.betas.shape[0] - 1

    @F.jit
    def forward_trajectory(self, rng_key : jax.Array, sample : Sample) -> Sample:
        """ Given an x_0, returns a trajectory of samples x_0, x_1, x_2, ..., x_T.
            Args:
                rng_key: The random key to use for the noise.
                sample: The initial sample x_0.
            
            Returns:
                A pytree of the same structure as sample, with a batch dimension of t+1.
                The 0th index corresponds to x_0, the 1st index corresponds to x_1, and so on.
        """
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        unfaltten_vmap = jax.vmap(unflatten)
        noise_flat = jax.random.normal(rng_key, (self.num_steps + 1,) + sample_flat.shape)
        # sum up the noise added at each step
        def scan_fn(prev_noise_accum, noise_beta):
            noise, alpha, beta = noise_beta
            noise_accum = jnp.sqrt(alpha)*prev_noise_accum + noise*jnp.sqrt(beta)
            return noise_accum, noise_accum
        noise_flat = jax.lax.scan(scan_fn, jnp.zeros_like(noise_flat[0]),
                                  (noise_flat, self.alphas, self.betas))[1]
        noisy_flat = noise_flat + jnp.sqrt(self.alphas_cumprod[:,None])*sample_flat[None,:]
        noise = unfaltten_vmap(noise_flat)
        noisy = unfaltten_vmap(noisy_flat)
        return noisy, noise

    # This will do the noising
    # forward process
    # will return noisy_sample, noise_eps, model_output 
    @F.jit
    def add_noise(self, rng_key : jax.Array, sample : Sample,
                  timestep : jax.Array) -> tuple[Sample, Sample, Sample]:
        """ Samples q(x_t | x_0). Returns a tuple containing (noisy_sample, noise, model_output).
        where model_output is based on the value of ``prediction_type``.

        Args:
            rng_key: The random key to use for the noise.
            sample: The initial sample x_0. Can be an arbitrary pytree.
            timestep: The timestep t to sample at.
        
        Returns:
            A tuple containing (noisy_sample, noise_epsilon, model_output).
            In the same structure as sample.
        """
        sqrt_alphas_prod = jnp.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alphas_prod = jnp.sqrt(1 - self.alphas_cumprod[timestep])
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape, dtype=sample_flat.dtype)
        noisy_flat = sqrt_alphas_prod * sample_flat + \
            sqrt_one_minus_alphas_prod*noise_flat
        noisy = unflatten(noisy_flat)
        noise = unflatten(noise_flat)
        if self.prediction_type == "epsilon":
            return noisy, noise, noise
        elif self.prediction_type == "sample":
            return noisy, noise, sample
        else:
            raise ValueError("Not supported prediction type")

    @F.jit
    def add_sub_noise(self, rng_key : jax.Array,
                      sub_sample : Sample, sub_timestep : jax.Array,
                      timestep : jax.Array) -> tuple[Sample, Sample, Sample]:
        """ Like add_noise, but assumes that sub_sample is x_{sub_timestep}
        rather than x_0. Note that timestep > sub_timestep or the behavior is undefined!
        """
        alphas_shifted = jnp.concatenate((jnp.ones((1,)), self.alphas_cumprod), axis=-1)
        alphas_prod = self.alphas_cumprod[timestep] / alphas_shifted[sub_timestep]
        sqrt_alphas_prod = jnp.sqrt(alphas_prod)
        sqrt_one_minus_alphas_prod = jnp.sqrt(1 - alphas_prod)
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sub_sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape, dtype=sample_flat.dtype)
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
        else:
            raise ValueError("Not supported prediction type")
    
    # returns E[x_0 | model_output, current sample]
    @F.jit
    def denoised_from_output(self, noised_sample : Sample, t : jax.Array, model_output : Sample) -> Sample:
        """ Returns E[x_0 | x_t] as computed by the model_output
        based on the value of ``prediction_type``
        """
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
        model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)
        if self.prediction_type == "epsilon":
            pred_sample = (sample_flat - beta_prod_t ** (0.5) * model_output_flat) / jnp.maximum(1e-6, alpha_prod_t ** (0.5))
        elif self.prediction_type == "sample":
            pred_sample = model_output_flat
        else:
            raise ValueError("Not supported prediction type")

        if self.clip_sample_range is not None:
            pred_sample = jnp.clip(pred_sample, -self.clip_sample_range, self.clip_sample_range)
        return unflatten(pred_sample)

    @F.jit
    def output_from_denoised(self, noised_sample : Sample, t : jax.Array, denoised_sample : Sample) -> Sample:
        """Returns the output a model should give given an x_t to denoise to x_0."""
        if self.prediction_type == "sample":
            return denoised_sample
        elif self.prediction_type == "epsilon":
            sqrt_alphas_prod = jnp.sqrt(self.alphas_cumprod[t])
            sqrt_one_minus_alphas_prod = jnp.sqrt(1 - self.alphas_cumprod[t])
            # noised_sample = sqrt_alphas_prod * denoised + sqrt_one_minus_alphas_prod * noise
            # therefore noise = (sample - sqrt_alphas_prod * denoised) / sqrt_one_minus_alphas_prod
            nosied_sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
            denoised_sample_flat, _ = jax.flatten_util.ravel_pytree(denoised_sample)
            noise = (nosied_sample_flat - sqrt_alphas_prod * denoised_sample_flat) / sqrt_one_minus_alphas_prod
            return unflatten(noise)
    
    @F.jit
    def compute_denoised(self, noised_sample : Sample, t : jax.Array, data_batch : Sample, data_mask : jax.Array = None) -> Sample:
        """Computes the true E[x_0 | x_t] given a batch of x_0's."""
        noised_sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
        data_batch_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(data_batch)
        # compute the mean
        sqrt_alphas_prod = jnp.sqrt(self.alphas_cumprod[t])
        one_minus_alphas_prod = 1 - self.alphas_cumprod[t]

        # forward diffusion equation for diffusing t timesteps:
        # noised_sample = sqrt_alphas_prod * denoised + sqrt_one_minus_alphas_prod * noise

        noise = (noised_sample_flat[None,:] - sqrt_alphas_prod * data_batch_flat)
        # the magnitude of the noise added
        noise_sqr = jnp.sum(noise**2, axis=-1)
        # p(x_t | x_0) prop exp(-1/2(x - mu)^2/sigma^2)
        log_likelihood = -0.5*noise_sqr / jnp.maximum(one_minus_alphas_prod, 1e-5)
        likelihood = jax.nn.softmax(log_likelihood, where=data_mask, initial=jnp.min(log_likelihood))
        likelihood = likelihood*data_mask if data_mask is not None else likelihood
        # # p(x_0 | x_t) = p(x_t | x_0) p(x_0) / p(x_t)
        # # where p(x_0) is uniform, so we effectively just to normalize the log likelihood
        # # over the x_0's
        # log_likelihood = log_likelihood - jax.scipy.special.logsumexp(log_likelihood, axis=0)
        # # log_likehood contains log p(x_0 | x_t) for all x_0's in the dataset

        # this is equivalent to the log-likelihood (up to a constant factor)
        denoised = jnp.sum(likelihood[:,None]*data_batch_flat, axis=0)
        return unflatten(denoised)

    # This does a reverse process step
    @F.jit
    def reverse_step(self, rng_key : jax.Array, sample : Sample,
                     timestep : jax.Array, delta_steps: jax.Array, model_output : Sample) -> Sample:
        """ Does a reverse step of the DDPM given a particular model output. Given x_t returns x_{t-delta_steps}. """
        chex.assert_trees_all_equal_shapes_and_dtypes(sample, model_output)
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)

        t = timestep
        prev_t = timestep - delta_steps
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        pred_sample = self.denoised_from_output(sample_flat, t, model_output_flat)

        pred_original_sample_coeff = jnp.sqrt(alpha_prod_t_prev) * beta_t / beta_prod_t
        current_sample_coeff = jnp.sqrt(alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_sample + current_sample_coeff * sample_flat

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t
        # variance = current_beta_t
        # we always take the log of variance, so clamp it to ensure it's not 0
        # variance = jnp.clip(variance, a_min=1e-20)
        sigma = jnp.sqrt(variance)
        noise = sigma*jax.random.normal(rng_key, pred_prev_sample.shape, pred_prev_sample.dtype)
        return unflatten(pred_prev_sample + noise)

    @F.jit
    def sample(self, rng_key : jax.Array, 
                    model : Callable[[jax.Array, Sample, jax.Array], Sample], 
                    sample_structure: Sample, 
                    *, num_steps : Optional[int] = None,
                    final_time : Optional[int] = None, trajectory : bool = False):
        """ Runs the reverse process, given a denoiser model, for a number of steps. """
        if final_time is None:
            final_time = 0
        if num_steps is None:
            num_steps = self.num_steps - final_time
        step_ratio = (self.num_steps - final_time) / num_steps
        # sample initial noise
        flat_structure, unflatten = tree.ravel_pytree_structure(sample_structure)
        random_sample = unflatten(jax.random.normal(rng_key, flat_structure.shape, flat_structure.dtype))

        if trajectory:
            # if we want to return the trajectory, do a scan.
            # num_steps must be a static integer then
            timesteps = (jnp.arange(0, num_steps + 1) * step_ratio).round()[::-1].astype(jnp.int32)
            curr_timesteps, prev_timesteps = timesteps[:-1], timesteps[1:]
            def step(carry, timesteps):
                rng_key, x_t = carry
                curr_T, prev_T = timesteps
                m_rng, s_rng, n_rng = jax.random.split(rng_key, 3)
                model_output = model(m_rng, x_t, curr_T)
                x_prev = self.reverse_step(s_rng, x_t, curr_T, curr_T - prev_T, model_output)
                return (n_rng, x_prev), x_prev
            carry, out = jax.lax.scan(step, (rng_key, random_sample), (curr_timesteps, prev_timesteps))
            _, sample = carry
            # add the initial noise to the front of the scan output
            out = jax.tree_map(lambda x, y: jnp.concatenate(
                                [jnp.expand_dims(x,axis=0), y], axis=0
                            ), random_sample, out)
            # reverse the trajectory along the time axis
            # so that traj[0] = x_0, traj[T] = x_T
            traj = jax.tree_map(lambda x: x[::-1, ...], out)
            return sample, traj
        else:
            static_loop = isinstance(num_steps, int)

            def do_step(carry, curr_T, prev_T):
                rng_key, x_t = carry
                m_rng, s_rng, n_rng = jax.random.split(rng_key, 3)
                model_output = model(m_rng, x_t, curr_T)
                x_prev = self.reverse_step(s_rng, x_t, curr_T, curr_T - prev_T, model_output)
                return (n_rng, x_prev)

            def loop_step(i, carry):
                curr_T = jnp.round((num_steps - i)*step_ratio).astype(jnp.int32)
                prev_T = jnp.round((num_steps - i - 1)*step_ratio).astype(jnp.int32)
                if not static_loop:
                    return jax.lax.cond(i < num_steps, 
                        do_step, lambda x,_a,_b: x, carry, curr_T, prev_T)
                else:
                    return do_step(carry, curr_T, prev_T)
            carry = (rng_key, random_sample)
            carry = jax.lax.fori_loop(0, self.num_steps if not static_loop else num_steps, loop_step, carry)
            _, sample = carry
            return sample

    def loss(self, rng_key : jax.Array, 
             model : Callable[[jax.Array, Sample, jax.Array], Sample],
             sample : Sample, t : Optional[jax.Array] = None, *,
             target_model : Callable[[jax.Array, Sample, jax.Array], Sample] | None = None,
             model_has_state_updates=False):
        """
        Computes the loss for the DDPM model.
        If t is None, a random t in [1, T] is chosen.
        """
        s_rng, t_rng, m_rng, tar_rng = jax.random.split(rng_key, 4)
        if t is None:
            t = jax.random.randint(t_rng, (), 0, self.num_steps) + 1
        noised_sample, _, target = self.add_noise(s_rng, sample, t)
        pred = model(m_rng, noised_sample, t)
        if model_has_state_updates:
            pred, state = pred
        if target_model is not None:
            target = target_model(tar_rng, noised_sample, t)
        chex.assert_trees_all_equal_shapes_and_dtypes(pred, target)

        pred_flat = jax.flatten_util.ravel_pytree(pred)[0]
        target_flat = jax.flatten_util.ravel_pytree(target)[0]
        loss = jnp.mean((pred_flat - target_flat)**2)

        if model_has_state_updates:
            return loss, state
        else:
            return loss
