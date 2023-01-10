import jax
import optax
import jax.numpy as jnp

from jinx.dataset.env import EnvDataset
from jinx.trainer import Trainer
from typing import NamedTuple, Any, Tuple
from functools import partial

from jinx.logging import logger

class ILSample(NamedTuple):
    x: jnp.array
    u: jnp.array

class ImitationLearning:
    def __init__(self, env, policy, net,
                    traj_length,
                    trajectories,
                    jac_lambda=0, **kwargs):
        # Dataset parameters
        self.env = env
        self.policy = policy
        self.net = net
        self.trajectories = trajectories
        self.traj_length = traj_length

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
        loss = u_loss
        stats = {}
        stats['u_loss'] = u_loss
        if self.jac_lambda > 0:
            pred_jac = jax.jacrev(apply_policy)(x)

            # ravel the trees!
            pred_jac, _ = jax.flatten_util.ravel_pytree(pred_jac)
            exp_jac, _ = jax.flatten_util.ravel_pytree(exp_jac)
            jac_loss = optax.safe_norm(pred_jac - exp_jac, 1e-5, ord=2)

            stats['jac_loss'] = jac_loss
            loss = loss + self.jac_lambda * jac_loss
        stats['loss'] = loss
        return loss, stats, fn_state

    def _map_fn(self, sample):
        xs, _ = sample
        exp_us = jax.vmap(self.policy)(xs)
        if self.jac_lambda > 0:
            exp_jacs = jax.vmap(jax.jacrev(self.policy))(xs)
        else:
            exp_jacs = None
        return xs, exp_us, exp_jacs

    def preprocess_data(self, dataset):
        dataset = dataset.map(self._map_fn)
        dataset = dataset.read()
        dataset = dataset.flatten()
        return dataset
    
    # Will return the trained policy
    def run(self, rng_key):
        rng_key, sk = jax.random.split(rng_key)

        dataset = EnvDataset(rng_key, self.env, self.policy, self.traj_length)

        # generate a random state to initialize
        # the network
        x0 = self.env.reset(rng_key)
        init_fn_params, init_fn_state = self.net.init(sk, x0)

        dataset = dataset[:self.trajectories]

        logger.info('il', "Training imitator neural network...")
        (fn_params, fn_state, _), _ = self.trainer.train(dataset, rng_key,
            init_fn_params=init_fn_params,
            init_fn_state=init_fn_state
        )
        final_policy = lambda x: self.net.apply(fn_params, fn_state, None, x)[0]
        return final_policy