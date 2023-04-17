import jax.numpy as jnp
import stanza.envs as envs
from jax.random import PRNGKey
import jax

pusht = envs.create('pusht')

x0 = pusht.reset(PRNGKey(0))

from stanza.envs.pusht import pretrained_net
net, params, state = pretrained_net()

key = PRNGKey(0)
nak, obsk = jax.random.split(key)
noised_action = jax.random.normal(nak, (16,2))
obs = jax.random.normal(obsk, (2,5))
timestep = 10*jnp.ones(())

expected = jnp.array([[-3.1368,  1.7825],
         [-4.0532, -0.1809],
         [-1.9427,  1.9806],
         [-4.8982,  1.8189],
         [ 0.7139,  1.2868],
         [-0.1256,  1.2424],
         [-2.1470,  2.5490],
         [-3.1863, -1.5174],
         [-1.7880,  0.2882],
         [ 1.7537,  3.2151],
         [-0.2146,  0.0532],
         [ 2.0484,  0.3218],
         [-0.4205,  2.0832],
         [ 1.9312, -0.8340],
         [-2.2321,  0.5938],
         [ 0.6338,  1.2663]])
res, _ = net.apply(params, state, None, noised_action,
                timestep, obs.reshape((-1,)))
print("single-sample success:", jnp.linalg.norm(res - expected) < 0.001)
# should be