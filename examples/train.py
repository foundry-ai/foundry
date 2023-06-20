import jax
from stanza.data import Data
from stanza.util.logging import logger
from jax.random import PRNGKey
import jax.numpy as jnp
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.ERROR, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

# A dataset of integers
dataset = Data.from_pytree(
    (jnp.arange(100), jnp.arange(100)[::-1])
)
sdataset = Data.from_pytree(
    (jnp.arange(10), jnp.arange(10))
)
sdataset = sdataset.shuffle(PRNGKey(42))
logger.info("Data shuffled {}", sdataset.data)
logger.info("Data batched {}", sdataset.batch(4).data.data)

logger.info("Dataset length: {}", dataset.length)
batches = dataset.batch(20)
logger.info("Batched length: {}", batches.length)

import haiku as hk

def net_fn(input):
    logger.info("Tracing model", only_tracing=True)
    input = jnp.atleast_1d(input)
    y = hk.nets.MLP([10, 1])(input)
    return jnp.squeeze(y, -1)

net = hk.transform(net_fn)
orig_init_params = net.init(PRNGKey(7), jnp.ones(()))

import optax
from stanza.util.random import permutation
import jax

logger.info("Permutation test: {}", permutation(PRNGKey(42), 10, n=6))

optimizer = optax.adamw(optax.cosine_decay_schedule(1e-3, 5000*10), 
                        weight_decay=1e-6)

def net_apply(params, rng_key, x):
    params = {m: {k: jnp.array(v) for (k,v) in sp.items()}
                for (m,sp) in params.items()}
    return net.apply(params, rng_key, x)

logger.info("Testing JIT apply")
jit_apply = jax.jit(net_apply)
jit_apply(orig_init_params, None, jnp.ones(()))
jit_apply(orig_init_params, None, jnp.ones(()))
jit_apply(orig_init_params, None, jnp.ones(()))
logger.info("Done testing JIT apply")

def loss_fn(params, _state, rng_key, sample):
    x, y = sample
    out = jit_apply(params, rng_key, x)
    loss = jnp.square(out - y)
    stats = {
        "loss": loss
    }
    return _state, loss, stats

from stanza import Partial
from stanza.train import Trainer
from stanza.train.rich import RichReporter
from stanza.train.wandb import WandbReporter
# import wandb
# wandb.init(project="train_test")

with WandbReporter() as wb:
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10, optimizer=optimizer)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            Partial(loss_fn), dataset,
            PRNGKey(42), init_params,
            hooks=[cb], jit=True
        )

logger.info("Training again...jit is cached so now training is fast")
with WandbReporter() as wb:
    with RichReporter(iter_interval=500) as cb:
        trainer = Trainer(epochs=5000, batch_size=10, optimizer=optimizer)
        init_params = net.init(PRNGKey(7), jnp.ones(()))
        res = trainer.train(
            Partial(loss_fn), dataset,
            PRNGKey(42), init_params,
            hooks=[cb], jit=True
        )