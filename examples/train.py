from stanza.data import Data, PyTreeData
from stanza.util.logging import logger

import jax

from jax.random import PRNGKey
import jax.numpy as jnp
import optax
import jax
import sys
import flax.linen as nn
from typing import Sequence
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.ERROR, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

from stanza.data.mnist import mnist
train_data, test_data = mnist()
def map_fn(sample):
   x, y = sample
   x = x.astype(jnp.float32) / 255.
   y = jax.nn.one_hot(y, 10)
   return x, y
train_data = PyTreeData.from_data(train_data.map(map_fn))
test_data = PyTreeData.from_data(test_data.map(map_fn))

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    x = jnp.atleast_1d(x)
    x = jnp.reshape(x, (-1,))
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return jax.nn.log_softmax(x)

model = MLP([128, 128, 10])

optimizer = optax.adamw(
    optax.cosine_decay_schedule(1e-3, 5000*10), 
    weight_decay=5e-3
)

def loss_fn(_state, params, rng_key, sample):
    x, y = sample
    log_probs = model.apply(params, x)
    loss = - jnp.sum(log_probs * y)

    target_class = jnp.argmax(y)
    predicted_class = jnp.argmax(log_probs)
    accuracy = 1.*(predicted_class == target_class)
    stats = {
        "loss": loss,
        "accuracy": accuracy
    }
    return _state, loss, stats

from stanza import Partial
from stanza.train import Trainer, batch_loss
from stanza.train.validate import Validator
from stanza.util.loop import every_kth_iteration, every_iteration, LoggerHook
from stanza.util.rich import ConsoleDisplay, StatisticsTable, LoopProgress
# import wandb
# wandb.init(project="train_test")

loss_fn = batch_loss(Partial(loss_fn))

display = ConsoleDisplay()
display.add("train", StatisticsTable(), interval=100)
display.add("train", LoopProgress(), interval=100)

from stanza.reporting.wandb import WandbDatabase
db = WandbDatabase("dpfrommer-projects/examples")
db = db.create()
logger.info(f"Logging to [blue]{db.name}[/blue]")

validator = Validator(PRNGKey(40), test_data, every_iteration)

from stanza.reporting.jax import JaxDBScope
db = JaxDBScope(db)
print_hook = LoggerHook(every_kth_iteration(100))

with display as w, db as db:
    trainer = Trainer(max_epochs=10, 
            batch_size=128, optimizer=optimizer)
    init_params = model.init(PRNGKey(7), train_data[0][0])
    logger_hook = db.statistic_logging_hook(
       log_cond=every_kth_iteration(1), buffer=100)
    res = trainer.train(
        train_data, loss_fn=loss_fn,
        rng_key=PRNGKey(42), init_params=init_params,
        train_hooks=[validator, w.train, 
                     logger_hook, print_hook], 
        jit=True
    )

# run manually and compare...
sys.exit(0)
optimizer = trainer.optimizer
from stanza.util.random import PRNGSequence
rng = PRNGSequence(42)
batch_size = 10
params = init_params
opt_state = optimizer.init(params)

def loss(params, batch):
   _, loss, stats = loss_fn(None, params, None, batch)
   return loss, stats

loss_grad = jax.jit(jax.grad(loss, argnums=0, has_aux=True))
for e in range(trainer.epochs):
    data = dataset.data
    idx = jax.random.permutation(next(rng), dataset.length)
    shuffled_data = jax.tree_map(
      lambda x: jnp.take(x, idx, axis=0, unique_indices=True),
      data)
    batched = jax.tree_map(
       lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
        shuffled_data
    )
    batches = dataset.length // 10
    for i in range(batches):
        batch = jax.tree_map(lambda x: x[i], batched)
        grad, stats = loss_grad(params, batch)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
    if e % 100 == 0:
        logger.info("{}", stats)