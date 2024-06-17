import jax
import jax.flatten_util
import jax.numpy as jnp
import stanza.util

from scipy.special import binom

def log_gaussian_kernel(x):
    x, _ = jax.flatten_util.ravel_pytree(x)
    return jnp.sum(-jnp.square(x))/2 - jnp.log(jnp.sqrt(2*jnp.pi))

def nadaraya_watson(data, kernel, h):
    xs, ys = data

    # Flatten the ys
    y0 = jax.tree_map(lambda x: x[0], ys)
    vf = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])
    ys = vf(ys)
    _, y_uf = jax.flatten_util.ravel_pytree(y0)

    def estimator(x):
        kernel_smoothing = lambda x, xi: kernel(jax.tree_map(lambda x,xi: (x-xi)/h, x,xi))
        log_smoothed = jax.vmap(kernel_smoothing, in_axes=[None, 0])(x, xs)
        smoothed = jax.nn.softmax(log_smoothed)
        y_est = jnp.sum(smoothed[:,None]*ys, axis=0)
        return y_uf(y_est)
    return estimator

def closest_diffuser(cond, data):
    x, y = data
    x = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(x)
    cond = jax.flatten_util.ravel_pytree(cond)[0]
    dists = jnp.sum(jnp.square(x-cond[None,...]), axis=-1)
    i = jnp.argmin(dists)
    closest = jax.tree_map(lambda x: x[i], y)

    @jax.jit
    def closest_diffuser(_, noised_value, t):
        return closest
    return closest_diffuser

def nw_cond_diffuser(cond, data, schedule, kernel, h):
    @jax.jit
    def diffuser(_, noised_value, t):
        sqrt_alphas_prod = jnp.sqrt(schedule.alphas_cumprod[t])
        one_minus_alphas_prod = 1 - schedule.alphas_cumprod[t]
        def comb_kernel(sample):
            x, y_hat_diff = sample
            x = jax.tree_map(lambda x: x/h, x)
            # Use one_minus_alphas_prod as the kernel bandwidth for the noised value
            y_diff = jax.tree_map(lambda x: x/one_minus_alphas_prod, y_hat_diff)
            return kernel(x) + log_gaussian_kernel(y_diff)
        x, y = data
        y_hat = jax.tree_map(lambda x: x*sqrt_alphas_prod, y)
        estimator_data = (x, y_hat), y
        estimator = nadaraya_watson(estimator_data, comb_kernel, h)
        return estimator((cond, noised_value))
    return diffuser