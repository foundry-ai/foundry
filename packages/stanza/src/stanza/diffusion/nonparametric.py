import jax
import jax.flatten_util
import jax.numpy as jnp
import stanza.util

from scipy.special import binom
from jax.scipy.special import factorial

#jax.config.update("jax_debug_nans", True)

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
            x = jax.tree.map(lambda x: x/h, x)
            y_diff = jax.tree.map(lambda x: x/one_minus_alphas_prod, y_hat_diff)
            return kernel(x) + log_gaussian_kernel(y_diff)
        x, y = data
        y_hat = jax.tree_map(lambda x: x*sqrt_alphas_prod, y)
        estimator_data = (x, y_hat), y
        estimator = nadaraya_watson(estimator_data, comb_kernel, h)
        return estimator((cond, noised_value))
    return diffuser

def expand_poly_features(degree, samples):
    # Adapted from sklearn.preprocessing.PolynomialFeatures
    num_features = samples.shape[-1]
    num_poly_features = binom(degree + samples.shape[-1], samples.shape[-1]).astype(int)
    poly_samples = jnp.empty((samples.shape[0], num_poly_features))
    poly_samples = poly_samples.at[:, 0].set(1)
    if degree >= 1:
        poly_samples = poly_samples.at[:, 1:1+num_features].set(samples)
    if degree >= 2:
        index = list(range(1, 1+num_features))
        current_col = 1 + num_features 
        index.append(current_col)
        for _ in range(2, degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(num_features):
                start = index[feature_idx]
                new_index.append(current_col)
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                # XP[:, start:end] are terms of degree d - 1
                # that exclude feature #feature_idx.
                poly_samples = poly_samples.at[:, current_col:next_col].set(
                    poly_samples[:, start:end] * samples[:, feature_idx : feature_idx + 1]
                )
                current_col = next_col

            new_index.append(current_col)
            index = new_index
    return poly_samples

def nw_local_poly(rng_key, data, schedule, degree, cond_kernel, noised_value_kernel, h_cond, h_noised, num_noised_samples):


    def estimator(cond, noised_value, t):

        xs, ys = data

        # Flatten the data
        vf = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])

        x0 = jax.tree_map(lambda x: x[0], xs)
        xs = vf(xs)
        _, x_uf = jax.flatten_util.ravel_pytree(x0)

        y0 = jax.tree_map(lambda x: x[0], ys)
        ys = vf(ys)
        _, y_uf = jax.flatten_util.ravel_pytree(y0)

        # Flatten input
        cond, cond_uf = jax.flatten_util.ravel_pytree(cond)
        noised_value, noised_value_uf = jax.flatten_util.ravel_pytree(noised_value)

        add_noise = lambda rng, x, y, t : (x, y, schedule.add_noise(rng, y, t)[0])
        batch_add_noise = lambda rng, x, y, t: jax.vmap(add_noise, in_axes=(0, None, None, None))(
            jax.random.split(rng_key, num_noised_samples), x, y, t
        ) 
        noised_samples = jax.vmap(batch_add_noise, in_axes=(0,0,0,None))(
            jax.random.split(rng_key, xs.shape[0]),
            xs,
            ys,
            t
        )
        xs = jnp.reshape(noised_samples[0], (-1, xs.shape[-1]))
        ys = jnp.reshape(noised_samples[1], (-1, ys.shape[-1]))
        yts = jnp.reshape(noised_samples[2], (-1, ys.shape[-1]))

        zs = jnp.concatenate((xs, yts), axis=-1)
        z = jnp.concatenate((cond, noised_value), axis=-1)
        
        cond_K = lambda x, xi: cond_kernel((x-xi)/h_cond)
        noised_value_K = lambda x, xi: noised_value_kernel((x-xi)/h_noised)
        
        log_prob = (
            jax.vmap(cond_K, in_axes=(None, 0))(cond, xs) + 
            jax.vmap(noised_value_K, in_axes=(None, 0))(noised_value, yts)
        )
        prob = jax.nn.softmax(log_prob)[:,None]

        poly_zs = expand_poly_features(degree, zs)
        poly_z = expand_poly_features(degree, z[None, :])

        M = poly_zs.T @ (prob * poly_zs)
        b = poly_zs.T @ (prob * ys)
        # M =   jnp.sqrt(prob) * poly_zs
        # b = jnp.sqrt(prob) * ys
        coeffs, _, _, _ = jnp.linalg.lstsq(M, b)

        y_est = jnp.dot(poly_z, coeffs)
        # jax.debug.print('{s}', s=jax.vmap(cond_K, in_axes=(None, 0))(cond, xs))
        # jax.debug.print('{s}', s=jax.vmap(noised_value_K, in_axes=(None, 0))(noised_value, yts))
        
        return noised_value_uf(y_est)
    
    return estimator

def nw_diffuser(cond, estimator):
    @jax.jit
    def diffuser(_, noised_value, t):
        return estimator(cond, noised_value, t)
    return diffuser

def comb(n, k):
    '''
    Returns n choose k.
    '''
    return factorial(n) / (factorial(n-k) * factorial(k))

def factorial2(n):
    '''
    Computes (n)!!, where n is an odd integer.
    '''
    n_half = (n-1)/2
    return factorial(n) / (factorial(n_half) * jnp.power(2,n_half))

def make_psi(degrees, Sigma, H):
    '''
    degrees: shape (d,) vector corresponding to [j_1,...,j_d]
    Sigma: shape (d,) vector of noise variances
    H: shape (d,) vector of bandwidth variances
    Returns the function psi.
    '''
    def psi(cond):
        dim = Sigma.shape[-1]
        Sigma_sqrt = jnp.sqrt(Sigma)
        Hinv = jnp.ones_like(H)/H
        M = Sigma * Hinv + jnp.ones_like(Sigma)
        Minv = jnp.ones_like(M)/M
        c = Hinv * Sigma_sqrt * cond
        K = jnp.power(2*jnp.pi, -dim/2) * jnp.sqrt( jnp.prod(Minv) * jnp.prod(Hinv)) \
            * jnp.exp(-0.5*(cond.T @ (Hinv * cond) - c.T @ (Minv * c)))  
        z = cond - Sigma_sqrt * Minv * c

        def exp_monomial(z, sigma, var, degree):

            def body_fn(i, carry):
                i = 2*i
                moment = jnp.pow(var, i//2) * factorial2(i-1) 
                poly_z = jnp.pow(z, degree-i)
                poly_sigma = jnp.pow(sigma, i)
                binom_coeff = comb(degree, i)
                return carry + moment * binom_coeff * poly_z * poly_sigma
                # log_moment = (i//2) * jnp.log(var) + jnp.log(factorial2(i-1))
                # poly_z = jnp.pow(z, degree-i)
                # log_poly_sigma = i*jnp.log(sigma)
                # log_binom_coeff = jnp.log(comb(degree, i))
                # return carry + jnp.exp(log_moment + log_binom_coeff + log_poly_sigma) * poly_z
            result = jnp.pow(z, degree) + jax.lax.fori_loop(1, (degree//2).astype(int) + 1, body_fn, 0)
            return result
        
        prod_monomials = 1000
        for i in range(dim):
            prod_monomials *= exp_monomial(z[i], Sigma[i], Minv[i], degrees[i])
        #jax.debug.print('{s}', s=prod_monomials)
        return K * prod_monomials
    
    return psi

def enumerate_degrees(num_features, degree):
    '''
    Returns an array of vectors of shape (num_features,) of nonnegative integers,
    each summing to at most `degree`.
    '''
    num_poly_features = binom(degree + num_features, num_features).astype(int)
    degree_vecs = jnp.empty((num_poly_features, num_features))
    degree_vecs = degree_vecs.at[0, :].set(jnp.zeros((num_features,)))

    features = jnp.eye(num_features)

    if degree >= 1:
        degree_vecs = degree_vecs.at[1:1+num_features, :].set(features)

    if degree >= 2:
        index = list(range(1, 1+num_features))
        current_row = 1 + num_features
        index.append(current_row)

        for _ in range(2, degree + 1):
            new_index = []
            end = index[-1]

            for feature_idx in range(num_features):
                start = index[feature_idx]
                new_index.append(current_row)
                next_row = current_row + end - start
                if next_row <= current_row:
                    break
                # XP[:, start:end] are terms of degree d - 1
                # that exclude feature #feature_idx.
                degree_vecs = degree_vecs.at[current_row:next_row, :].set(
                    degree_vecs[start:end, :] + features[feature_idx : feature_idx + 1, :]
                )
                current_row = next_row

            new_index.append(current_row)
            index = new_index

    return degree_vecs

def nw_local_poly_closed(data, schedule, degree, h2):

    def estimator(cond, noised_value, t):
        xs, ys = data

        # Flatten the data
        vf = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])

        x0 = jax.tree_map(lambda x: x[0], xs)
        xs = vf(xs)
        _, x_uf = jax.flatten_util.ravel_pytree(x0)

        y0 = jax.tree_map(lambda x: x[0], ys)
        ys = vf(ys)
        _, y_uf = jax.flatten_util.ravel_pytree(y0)

        # Flatten input
        cond, cond_uf = jax.flatten_util.ravel_pytree(cond)
        noised_value, noised_value_uf = jax.flatten_util.ravel_pytree(noised_value)

        H = h2*jnp.ones((xs.shape[-1] + ys.shape[-1],))
    
        z = jnp.concatenate((cond, noised_value), -1)
        alphas_prod = schedule.alphas_cumprod[t]
        zs = jnp.concatenate((xs, jnp.sqrt(alphas_prod)*ys), -1)
        num_features = z.shape[-1]
        Sigma = jnp.zeros((xs.shape[-1],))
        Sigma = jnp.append(
            Sigma,
            jnp.repeat(jnp.array([1-alphas_prod]), ys.shape[-1])
        )
        #jax.debug.print('{s}', s=Sigma)

        degree_vecs = enumerate_degrees(num_features, degree)

        # such that: degree_matrix[i, j, :] = degree_vecs[i, :] + degree_vecs[j, :]
        degree_matrix = degree_vecs[None, :, :] + degree_vecs[:, None, :]
        apply_psi = lambda deg_vec: jnp.sum(
            jax.vmap(make_psi(deg_vec, Sigma, H))(zs - z),
            axis=0
        )
        A = jax.vmap(jax.vmap(apply_psi))(degree_matrix)
        #jax.debug.print('{s}', s=A)

        
        apply_psi = lambda deg_vec: jnp.sum(
            jax.vmap(make_psi(deg_vec, Sigma, H))(zs - z)[:, None] * ys,
            axis=0
        )
        b = jax.vmap(apply_psi)(degree_vecs)
        #jax.debug.print('{s}', s=b)

        
        coeffs, _, _, _ = jnp.linalg.lstsq(A, b)

        poly_z = expand_poly_features(degree, z[None, :])
        y_est = jnp.dot(poly_z, coeffs)

        # jax.debug.print('{s}', s=estimate.shape)
        
        return noised_value_uf(y_est)
    
    return estimator


