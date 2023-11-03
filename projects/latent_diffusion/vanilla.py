import optax
import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass
from stanza.runtime import activity

from stanza import partial
from stanza.train import batch_loss, express as express_trainer
from stanza.train.validate import Generator
from stanza.util.random import PRNGSequence
import stanza.util.loop as loop
from stanza.reporting import Image

from stanza.diffusion.ddpm import DDPMSchedule
from stanza.util.logging import logger

from latent_diffusion.nets import make_network
from latent_diffusion.data import load_data

@dataclass(jax=True)
class Config:
    dataset : str = "cifar10"

    learning_rate : float = 1e-3
    epochs : int = 50
    batch_size : int = 16
    timesteps : int = 100

    rng_seed : int = 42

def loss_fn(config, schedule, net,
            fn_state, fn_params, 
            rng_key, sample):
    rng = PRNGSequence(rng_key)
    timestep = jax.random.randint(next(rng), (), 0, schedule.num_steps)
    noisy, _, target = schedule.add_noise(next(rng), sample, timestep)
    pred = net.apply(fn_params, noisy, train=True)

    loss = jnp.mean(jnp.square(target - pred))
    stats = { "loss": loss }
    return fn_state, loss, stats


@activity(Config)
def train(config, db):
    rng = PRNGSequence(config.rng_seed)
    exp = db.create()

    data = load_data(config.dataset, next(rng))
    # make the network
    data_sample = data.train_data[0]
    net, params = make_network(next(rng), data_sample)

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps,
        clip_sample_range=1.
    )
    loss = batch_loss(partial(loss_fn, config, schedule, net))

    steps = config.epochs * (data.train_data.length // config.batch_size)
    optimizer = optax.adamw(
        optax.cosine_decay_schedule(config.learning_rate, steps),
    )

    def do_generate(train_state, rng_key):
        fn_params = train_state.fn_params
        model = lambda rng, sample, t: net.apply(
            fn_params, sample, timestep=t, rngs={"dropout":rng}
        )
        sample = schedule.sample(rng_key, model, data_sample)
        img = (255.*((sample+1)/2)).astype(jnp.uint8)
        return img
    
    generator = Generator(
        bucket=exp,
        rng_key=next(rng),
        samples=16,
        generate_fn=do_generate,
        visualizer=data.samples_visualizer
    )
    trainer = express_trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        max_epochs=config.epochs,
        validate_condition=loop.every_epoch,
        validate_dataset=data.test_data,
        validate_batch_size=config.batch_size,
        validate_rng=next(rng),
        bucket=exp,
        loss_fn=loss,
        train_hooks=[generator]
    )
    res = trainer.train(
        data.train_data,
        init_params=params,
        rng_key=next(rng)
    )