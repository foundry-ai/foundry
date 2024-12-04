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

import logging
logger = logging.getLogger(__name__)

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
        return self.encoder(x)[0]

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


def train_embedding(train_data, mode="vae",
                    iterations=4096):
    latents = None
    if mode == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(max_iter=500, perplexity=30)
        logger.info("Learning T-SNE embedding...")
        latents = tsne.fit_transform(train_data.reshape((train_data.shape[0], -1)))
        latents = npx.array(latents)
        latents = latents / npx.std(latents, axis=0)
        logger.info("Embedding done!")
    train_data = PyTreeData((train_data,latents))
    ae = Autoencoder()
    ae_vars = ae.init(foundry.random.key(42), npx.zeros((28, 28, 1), npx.float32), foundry.random.key(42))
    @F.jit
    def loss(vars, rng_key,  sample):
        x, z_guidance = sample
        z, stddev, x_hat = ae.apply(vars, x, rng_key)
        log_likelihood = -npx.sum(npx.square(x - x_hat))
        kl = kl_gaussian(z, npx.square(stddev))
        elbo = log_likelihood - kl
        if latents is not None:
            guidance_loss = 10*npx.sum(npx.square(z - z_guidance))
        else:
            guidance_loss = 0.
        loss = -elbo + guidance_loss
        return foundry.train.LossOutput(
            loss=loss,
            metrics=dict(
                log_likelihood=log_likelihood,
                kl=kl,
                guidance=guidance_loss,
                loss=loss,
            ),
        )
    batch_loss = foundry.train.batch_loss(loss)
    opt = optax.adamw(optax.cosine_decay_schedule(1e-3, iterations))
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
                if step.iteration % 1000 == 0:
                    foundry.train.console.log(step.iteration, metrics)
    
    @F.jit
    def encode(x):
        return ae.apply(ae_vars, x, method=Autoencoder.encode)
    @F.jit
    def decode(z):
        return ae.apply(ae_vars, z, method=Autoencoder.decode)
    
    # re-encode the data using the trained model
    latents = jax.vmap(encode)(train_data.as_pytree()[0])
    return latents, encode, decode