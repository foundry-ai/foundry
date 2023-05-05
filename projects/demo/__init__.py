from stanza.runtime import activity
from stanza.util.dataclasses import dataclass
from stanza.util.logging import logger

from stanza.data import Data
from jax.random import PRNGKey
from functools import partial

from stanza.train import Trainer
from stanza.train.rich import RichReporter
from stanza.train.ema import EmaHook
from stanza.util.random import permutation

import jax.numpy as jnp
import haiku as hk
import optax

@dataclass(frozen=True)
class Config:
    seed: int = 42
    param: float = 0.1
    name: str = "foo"

def net_fn(input):
    input = jnp.atleast_1d(input)
    y = hk.nets.MLP([10, 1])(input)
    return jnp.squeeze(y, -1)

def loss_fn(net, params, rng_key, sample):
    x, y = sample
    out = net.apply(params, rng_key, x)
    loss = jnp.square(out - y)

    stats = {
        "loss": loss
    }
    return loss, stats

@activity(Config)
def train(config, database):
    logger.info("Parsed config: {}", config)
    dataset = Data.from_pytree(
        (jnp.arange(100), jnp.arange(100)[::-1])
    )
    logger.info("Dataset length: {}", dataset.length)
    batches = dataset.batch(20)
    logger.info("Batched length: {}", batches.length)
    logger.info("Permutation test: {}", permutation(PRNGKey(42), 10, n=6))

    net = hk.transform(net_fn)

    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                5000*10, alpha=0.1)),
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-5e-3)
    )
    loss = partial(loss_fn, net)

    ema = EmaHook()
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            loss, dataset,
            PRNGKey(42), init_params,
            hooks=[ema,cb]
        )