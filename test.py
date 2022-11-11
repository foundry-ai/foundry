from athena.dataset import Dataset
import jax.numpy as jnp
import time



data = Dataset.from_pytree(jnp.ones((10*256*16, 5)))

def collect(sum, item):
    return sum + item

begin = time.time()
print("Doing streaming fold")
res = data.iter().stream_fold(collect, jnp.zeros(5,), vsize=256)
print(res)
print(time.time() - begin)
begin = time.time()
res = data.iter().fold(collect, jnp.zeros(5,))
print(res)
print(time.time() - begin)
assert res.shape == (5,)
assert jnp.all(res == 5000)