import jax
import optax
import jax.numpy as jnp

from stanza.dataset.env import EnvDataset
from stanza.trainer import Trainer
from typing import NamedTuple, Any, Tuple

from stanza.util.logging import logger

class ILSample(NamedTuple):
    x: jnp.array
    u: jnp.array

class ImitationLearning:
    # Either specify net and pass in a dataset
    # at train time, or pass in env, policy, traj_length, trajectories
    def __init__(self, net,
                 jac_lambda=0, **kwargs):
        # Dataset parameters
        self.net = net
        self.jac_lambda = jac_lambda

        self.trainer = Trainer(
            self._loss_fn,
            preprocess=self.preprocess_data,
            **kwargs
        )

    def _loss_fn(self, fn_params, fn_state, rng_key, sample):
        x, exp_u, exp_jac = sample
        pred_u, fn_state = self.net.apply(fn_params, fn_state, rng_key, x)

        apply_policy = lambda x: self.net.apply(fn_params, fn_state, rng_key, x)[0]

        u_loss = optax.safe_norm(pred_u - exp_u, 1e-5, ord=2)
        # u_loss = jnp.sum((pred_u - exp_u)**2)
        loss = u_loss
        stats = {}
        stats['u_loss'] = u_loss
        if self.jac_lambda > 0:
            pred_jac = jax.jacrev(apply_policy)(x)

            # ravel the trees!
            pred_jac, _ = jax.flatten_util.ravel_pytree(pred_jac)
            exp_jac, _ = jax.flatten_util.ravel_pytree(exp_jac)
            jac_loss = optax.safe_norm(pred_jac - exp_jac, 1e-5, ord=2)
            jac_loss = jnp.sum((pred_jac - exp_jac)**2)

            stats['jac_loss'] = jac_loss
            loss = loss + self.jac_lambda * jac_loss
        stats['loss'] = loss
        return fn_state, loss, stats

    def _map_fn(self, sample):
        xs, us, jacs = (sample  + (None,)) if len(sample) == 2 else sample
        # remove last state for the xs
        xs = jax.tree_util.tree_map(lambda x: x[:-1], xs)
        return xs, us, jacs

    def preprocess_data(self, dataset):
        dataset = dataset.map(self._map_fn)
        dataset = dataset.read()
        dataset = dataset.flatten()
        return dataset
    
    # Will return the trained policy
    def run(self, rng_key, dataset):
        rng_key, sk = jax.random.split(rng_key)

        x0 = jax.tree_util.tree_map(lambda x: x[0], dataset.get(dataset.start)[0])

        # generate a random state to initialize
        # the network
        init_fn_params, init_fn_state = self.net.init(sk, x0)


        logger.info('il', "Training imitator neural network...")
        res = self.trainer.train(dataset, rng_key,
            init_fn_params=init_fn_params,
            init_fn_state=init_fn_state
        )
        final_policy = lambda x: self.net.apply(res.fn_params, res.fn_state, None, x)[0]
        return final_policy