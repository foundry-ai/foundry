from stanza.data import Data
from stanza.util.logging import logger
from jax.random import PRNGKey
import jax.numpy as jnp

# A dataset of integers
dataset = Data.from_pytree(
    (jnp.arange(100), jnp.arange(100)[::-1])
)
logger.info("Dataset length: {}", dataset.length)
batches = dataset.batch(20, ret_first=False)
logger.info("Batched length: {}", batches.length)

import haiku as hk

def net_fn(input):
    input = jnp.atleast_1d(input)
    y = hk.nets.MLP([10, 1])(input)
    return jnp.squeeze(y, -1)

net = hk.transform(net_fn)

import optax
from stanza.train import Trainer
from stanza.train.rich import RichReporter

optimizer = optax.chain(
    # Set the parameters of Adam. Note the learning_rate is not here.
    optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                            5000*10, alpha=0.1)),
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    # Put a minus sign to *minimise* the loss.
    optax.scale(-5e-3)
)

def loss_fn(params, rng_key, sample):
    x, y = sample
    out = net.apply(params, rng_key, x)
    loss = jnp.square(out - y)

    stats = {
        "loss": loss
    }
    return loss, stats

with RichReporter(iter_interval=500) as cb:
    trainer = Trainer(epochs=5000)
    init_params = net.init(PRNGKey(7), jnp.ones(()))
    res = trainer.train(
        loss_fn, dataset,
        PRNGKey(42), init_params,
        hooks=[cb]
    )