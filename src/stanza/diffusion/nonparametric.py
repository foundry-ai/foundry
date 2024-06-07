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

def nw_cond_diffuser(cond, data, schedule, kernel, h):
    @jax.jit
    def diffuser(_, noised_value, t):
        def comb_kernel(sample):
            x, y = sample
            # Use one_minus_alphas_prod as the kernel for the noised value
            one_minus_alphas_prod = 1 - schedule.alphas_cumprod[t]
            y = jax.tree_map(lambda x: x/one_minus_alphas_prod, y)
            return kernel(x) + log_gaussian_kernel(y)
        x, y = data
        estimator_data = (x, y), y
        estimator = nadaraya_watson(estimator_data, comb_kernel, h)
        return estimator((cond, noised_value))
    return diffuser