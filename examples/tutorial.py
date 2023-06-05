import jax
import jax.numpy as jnp
import haiku as hk
import optax

a = jnp.array([0., 1., 2.])

def net(x : jnp.array):
    net = hk.nets.MLP([10, 10, 10])
    y = net(x)
    linear = hk.Linear(10, name='final')
    return jnp.sum(linear(y))

haiku_net = hk.transform(net)
params = haiku_net.init(jax.random.PRNGKey(42), a)

params_flat, unflatten = jax.flatten_util.ravel_pytree(params)
params_flat = params_flat*2
print(params_flat.shape)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state):
    grad_apply = jax.grad(haiku_net.apply, argnums=0)
    grad = grad_apply(params, None, a)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

params, opt_state = step(params, opt_state)
params, opt_state = step(params, opt_state)
params, opt_state = step(params, opt_state)
params, opt_state = step(params, opt_state)
params, opt_state = step(params, opt_state)
# print(grad)