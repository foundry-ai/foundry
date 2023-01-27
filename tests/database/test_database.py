from stanza.dataset import Dataset
import jax.numpy as jnp

# def test_simple():
#     data = Dataset.from_pytree({'a': jnp.zeros((5, 3,2)), 'b': jnp.ones((5, 5))})
#     assert data.length == 5
#     i = 0
#     for x in data:
#         assert x['a'].shape == (3,2)
#         assert x['b'].shape == (5,)
#         assert not x['a'].any()
#         assert jnp.all(x['b'] == 1)
#         i = i + 1
#     assert i == 5

# def test_stream():
#     data = Dataset.from_pytree({'a': jnp.zeros((500, 3,2)), 'b': jnp.ones((500, 5))})
#     stream = data.stream()

#     def collect(sum, item):
#         return sum + item['b']
    
#     res = stream.fold(collect, jnp.zeros(5,))
#     assert res.shape == (5,)
#     assert jnp.all(res == 500)