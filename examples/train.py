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

from stanza.datasets.mnist import mnist
train_data, test_data = mnist(splits=("train", "test"))
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
from stanza.util.rich import ConsoleDisplay, StatisticsTable, LoopProgressBar

loss_fn = batch_loss(Partial(loss_fn))

display = ConsoleDisplay()
display.add(StatisticsTable(), interval=100)
display.add(LoopProgressBar(), interval=100)

trainer = Trainer(max_epochs=10, 
        batch_size=128, optimizer=optimizer)
init_params = model.init(PRNGKey(7), train_data[0][0])
res = trainer.train(
    train_data, loss_fn=loss_fn,
    rng_key=PRNGKey(42), init_params=init_params,
    train_hooks=[display], jit=True
)