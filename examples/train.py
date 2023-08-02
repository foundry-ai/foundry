import jax
from stanza.data import Data
from stanza.util.logging import logger
from jax.random import PRNGKey
import jax.numpy as jnp
import optax
import jax
import flax.linen as nn
from typing import Sequence
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


class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    x = jnp.atleast_1d(x)
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

model = MLP([10, 1])

orig_init_params = model.init(PRNGKey(7), jnp.ones(()))

optimizer = optax.adamw(
    optax.cosine_decay_schedule(1e-3, 5000*10), 
    weight_decay=1e-6
)

def net_apply(params, rng_key, x):
    y = model.apply(params, x)
    y = jnp.squeeze(y, -1)
    return y

logger.info("Testing JIT apply")
jit_apply = jax.jit(net_apply)
jit_apply(orig_init_params, PRNGKey(42), jnp.ones(()))
logger.info("Done testing JIT apply")

def loss_fn(_state, params, rng_key, sample):
    x, y = sample
    out = jit_apply(params, rng_key, x)
    loss = jnp.square(out - y)
    stats = {
        "loss": loss
    }
    return _state, loss, stats

from stanza import Partial
from stanza.train import Trainer, batch_loss
from stanza.reporting.jax import log_every_kth_iteration
from stanza.util.rich import ConsoleDisplay, StatisticsTable, LoopProgress
# import wandb
# wandb.init(project="train_test")

loss_fn = batch_loss(Partial(loss_fn))

display = ConsoleDisplay()
display.add("train", StatisticsTable(), interval=100)
display.add("train", LoopProgress(), interval=100)

from stanza.reporting.wandb import WandbDatabase
db = WandbDatabase("dpfrommer-projects/examples")
db = db.open("train")

from stanza.reporting.jax import JaxDBScope
db = JaxDBScope(db)

with display as w, db as db:
    trainer = Trainer(epochs=5000, batch_size=10, optimizer=optimizer)
    init_params = model.init(PRNGKey(7), jnp.ones(()))

    wb_logger = db.log_hook(
       log_cond=log_every_kth_iteration(100),
       buffer=100)
    res = trainer.train(
        loss_fn, dataset,
        PRNGKey(42), init_params,
        hooks=[w.train, wb_logger], jit=True
    )
with display as w:
    trainer = Trainer(epochs=5000, batch_size=10, optimizer=optimizer)
    init_params = model.init(PRNGKey(7), jnp.ones(()))
    res = trainer.train(
        loss_fn, dataset,
        PRNGKey(42), init_params,
        hooks=[w.train], jit=True
    )