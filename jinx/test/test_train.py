from jinx.train import Schedule, Quantity
from jinx.dataset import Dataset

import jax.numpy as jnp

def test_trainer():
    def loss_fn(vars, sample):
        return vars['a']*sample**2

    def test_fn(vars, sample):
        return vars['a']

    train_data = Dataset.from_pytree({'a': jnp.zeros((500, 3,2)), 'b': jnp.ones((500, 5))})
    test_data = Dataset.from_pytree({'a': jnp.zeros((500, 3,2)), 'b': jnp.ones((500, 5))})

    s = Schedule(Quantity.epochs(100))
    params = s.init_params('vars')
    opt = s.init_optimizer('optim')

    train_data = s.init_data('train_data', train_data)
    test_data = s.init_data('test_data', test_data)

    s.train(params, train_data, loss_fn, opt, Quantity.epochs(1))
    s.test(params, test_data, test_fn, Quantity.epochs(1))