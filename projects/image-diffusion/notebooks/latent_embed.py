import foundry.random
import foundry.numpy as npx
import foundry.train

import foundry.core as F
import flax.linen as nn
import flax.linen.activation as activations

import math
import jax
import optax
import foundry.train.console

from foundry.data import PyTreeData
from typing import Sequence

class Encoder(nn.Module):
    latent_dim: int = 2

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1)
        for i in [512, 256, 128, 128, 64]:
            x = nn.Dense(i)(x)
            x = activations.relu(x)
        mean = nn.Dense(self.latent_dim)(x)
        log_stddev = nn.Dense(self.latent_dim)(x)
        stddev = npx.exp(log_stddev)
        return mean, stddev

class Decoder(nn.Module):
    features = [32, 16, 1]
    image_shape: Sequence[int] = (28, 28, 1)

    @nn.compact
    def __call__(self, z):
        x = z
        for i in [64, 128, 128, 256, 512]:
            x = nn.Dense(i)(x)
            x = activations.relu(x)
        x = nn.Dense(math.prod(self.image_shape))(x)
        x = x.reshape(self.image_shape)
        return x

class Autoencoder(nn.Module):
    latent_dim: int = 2
    image_shape: Sequence[int] = (28, 28, 1)

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.image_shape)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x, rng_key):
        z, stddev = self.encoder(x)
        z = z + stddev * foundry.random.normal(rng_key, z.shape)
        return z, stddev, self.decoder(z)


def kl_gaussian(mean: jax.Array, var: jax.Array) -> jax.Array:
    mean, var = mean.reshape((-1)), var.reshape((-1))
    return 0.5 * npx.sum(
        -npx.log(var) - 1.0 + var + npx.square(mean), 
        axis=-1
    )


def train_embedding(train_data, mode="vae"):
    latent = None
    if mode == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_iter=300)
        print("Learning embedding...")
        tsne_results = tsne.fit_transform(train_data)
        print("Embedding done!")
    train_data = PyTreeData((train_data,latent))
    ae = Autoencoder()
    ae_vars = ae.init(foundry.random.key(42), npx.zeros((28, 28, 1), npx.float32), foundry.random.key(42))
    @F.jit
    def loss(vars, rng_key,  sample):
        x, z = sample
        if mode == "vae":
            z, stddev, x_hat = ae.apply(vars, x, rng_key)
            log_likelihood = -npx.sum(npx.square(x - x_hat))
            kl = kl_gaussian(z, npx.square(stddev))
            elbo = log_likelihood - kl
            loss = -elbo
            return foundry.train.LossOutput(
                loss=loss,
                metrics=dict(
                    log_likelihood=log_likelihood,
                    kl=kl,
                    loss=loss,
                ),
            )
        else:
            x_hat = ae.apply(vars, z, method=Autoencoder.decode)
            loss = npx.sum(npx.square(x - x_hat))
            return foundry.train.LossOutput(
                loss=loss,
                metrics=dict(
                    loss=loss,
                ),
            )
    batch_loss = foundry.train.batch_loss(loss)

    iterations = 1024*32
    opt = optax.adam(optax.cosine_decay_schedule(3e-4, iterations))
    ae_opt_state = opt.init(ae_vars["params"])

    with foundry.train.loop(
        train_data.stream().batch(256),
        iterations=iterations,
        rng_key=foundry.random.key(42),
    ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                ae_opt_state, ae_vars, metrics = foundry.train.step(
                    batch_loss, opt, ae_opt_state, ae_vars, step.rng_key, step.batch
                )
                if step.iteration % 1024 == 0:
                    foundry.train.console.log(step.iteration, metrics)
    
    @F.jit
    def encode(x):
        return ae.apply(ae_vars, x, method=Autoencoder.encode)
    @F.jit
    def decode(z):
        return ae.apply(ae_vars, z, method=Autoencoder.decode)
    return encode, decode