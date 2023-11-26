import optax
import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass
from stanza.runtime import activity

from stanza import partial
from stanza.data import PyTreeData
from stanza.train import batch_loss, express as express_trainer, Trainer, SAMTrainer
from stanza.train.validate import Generator, Validator
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
    batch_size : int = 64
    timesteps : int = 100

    measure_sharpness : bool = False

    rng_seed : int = 42
    rho : float = 0.

def loss_fn(config, schedule, normalizer, net,
            fn_state, fn_params, 
            rng_key, sample):
    rng = PRNGSequence(rng_key)
    timestep = jax.random.randint(next(rng), (), 0, schedule.num_steps)
    normalized_sample = normalizer.normalize(sample)
    noisy, _, target = schedule.add_noise(next(rng), normalized_sample, timestep)
    pred = net.apply(fn_params, noisy, timestep=timestep, train=True)
    loss = jnp.mean(jnp.square(target - pred))
    stats = { "loss": loss }
    return fn_state, loss, stats


@activity(Config)
def train(config, db):
    rng = PRNGSequence(config.rng_seed)
    exp = db.create()

    data = load_data(config.dataset, next(rng))
    normalizer = data.normalizer
    # make the network
    data_sample = data.train_data[0]
    net, params = make_network(next(rng), data_sample)

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps,
        clip_sample_range=1.
    )
    loss = batch_loss(partial(loss_fn, config, schedule, normalizer, net))

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
        sample = normalizer.unnormalize(sample)
        return sample

    # log some ground truth samples
    test_samples = PyTreeData.from_data(data.test_data[:data.n_visualize_samples]).data
    exp.log({"test_samples": data.samples_visualizer(test_samples)})
    
    generator = Generator(
        bucket=exp,
        rng_key=next(rng),
        samples=data.n_visualize_samples,
        generate_fn=do_generate,
        visualizer=data.samples_visualizer
    )
    hooks = [generator]

    def measure_sharpness(_, params, rng_key, batch):
        def noise_sample(sample, rng_key):
            a, b = jax.random.split(rng_key)
            timestep = jax.random.randint(a, (), 0, schedule.num_steps)
            normalized_sample = normalizer.normalize(sample)
            noisy, _, target = schedule.add_noise(b, normalized_sample, timestep)
            return noisy, timestep, target
        data_rng, loss_rng = jax.random.split(rng_key)
        rngs = jax.random.split(data_rng, batch.shape[0])
        noisy, timesteps, target = jax.vmap(noise_sample)(batch, rngs)
        rngs = jax.random.split(loss_rng, batch.shape[0])
        def loss_fn(params):
            model = lambda rng, sample, t: net.apply(
                params, sample, timestep=t, rngs={"dropout":rng}, train=False
            )
            pred = jax.vmap(model)(rngs, noisy, timesteps)
            return jnp.mean(jnp.square(target - pred))
        grad = jax.grad(loss_fn)(params)
        global_norm = optax.global_norm(grad)
        grad = jax.tree_map(lambda x: x / global_norm, grad)
        loss = loss_fn(params)
        eta = 0.01
        perturbed_params = jax.tree_map(lambda p, g: p + eta * g, params, grad)
        p_loss = loss_fn(perturbed_params)
        l = (p_loss - loss)/eta
        return {"value": l}
    
    if config.measure_sharpness:
        validator = Validator(
            rng_key=next(rng),
            dataset=data.test_data,
            batch_size=config.batch_size,
            condition=loop.every_epoch,
            stat_key="sharpness",
            stat_fn=measure_sharpness,
        )
        hooks.append(validator)

    sub_optimizer = optax.scale(config.rho) if config.rho > 0. else None
    sam_args = {
        "sub_optimizer": sub_optimizer,
        "cls": SAMTrainer
    } if config.rho > 0. else {}

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
        train_hooks=hooks,
        **sam_args
    )
    res = trainer.train(
        data.train_data,
        init_params=params,
        rng_key=next(rng)
    )