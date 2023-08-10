# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""


import time
import itertools
import wandb

import numpy.random as npr

import optax
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from stanza.data.mnist import mnist

def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    loss = -jnp.mean(jnp.sum(preds * targets, axis=1))
    return loss

def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

import flax.linen as nn
from typing import Sequence

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return jax.nn.log_softmax(x)

model = MLP([250, 100, 10])
predict = model.apply

if __name__ == "__main__":
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    train, test = mnist()
    train_images, train_labels = train.data
    test_images, test_labels = test.data

    train_images = jnp.reshape(train_images, (train_images.shape[0], -1)) / 255.
    test_images = jnp.reshape(test_images, (test_images.shape[0], -1)) / 255.
    test_labels = jax.nn.one_hot(test_labels, 10)
    train_labels = jax.nn.one_hot(train_labels, 10)

    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    optimizer = optax.adamw(
        optax.cosine_decay_schedule(1e-3, 5000*10), 
        weight_decay=5e-3
    )

    @jit
    def update(i, opt_state, params, batch):
        def l(params, batch):
            ls = loss(params, batch)
            return ls, ls
        grads, loss_val = grad(l, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    init_params = model.init(rng, jnp.ones((1, 28 * 28)))
    params = init_params
    opt_state = optimizer.init(init_params)
    itercount = itertools.count()

    wandb.init()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            params, opt_state, train_loss = update(
                next(itercount), opt_state, params, next(batches)
            )
            train_acc = accuracy(params, (train_images, train_labels))
            test_loss = loss(params, (test_images, test_labels))
            test_acc = accuracy(params, (test_images, test_labels))
            wandb.log({"train_acc": train_acc, "test_acc": test_acc,
                       "train_loss": train_loss, "test_loss": test_loss})
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")